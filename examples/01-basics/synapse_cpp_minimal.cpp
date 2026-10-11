// synapse_cpp_minimal.cpp -- the smallest correct SynapseAI C++ program that runs a GEMM
// on Gaudi 2.
//
// Adapted from a verified, working probe (`Archive/by-date/2026-10-11/synapse_cpp/probe.cpp`
// in the source project). The API calls below are the ones that were confirmed to build and
// run against Synapse 1.24.1; the surrounding harness is trimmed to the minimum. It has NOT
// been re-run in this manual's repository -- treat it as an illustrative skeleton, not as a
// tested artifact.
//
// Build (verified against Synapse 1.24.1):
//   g++ -O2 -std=c++17 synapse_cpp_minimal.cpp \
//       -I/usr/include/habanalabs -L/usr/lib/habanalabs -lSynapse -o synapse_cpp_minimal
//
// Note the library name: `libSynapse.so`, capital S. There is no `libsynapse.so`, and the
// graph compiler is statically linked into it -- there is no separate `libgc.so`.
//
// What to look for in the output:
//   * `synNodeCreate(guid="gemm") -> synSuccess`   -- the GC accepts the node type
//   * `synGraphCompile -> synSuccess`              -- lowering succeeded
//   * `OK` at the end                              -- launch + sync succeeded
//
// Run it on an IDLE module. A vendor tool shows an idle Gaudi 2 at a low memory baseline; a
// device with a large footprint belongs to another process.

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>

#include <synapse_api.h>
#include <synapse_common_types.hpp>   // REQUIRED: synTensorDescriptor is only forward-declared
                                      // in the C header that synapse_api.h pulls in.

#define CHECK(call)                                                            \
    do {                                                                       \
        synStatus _st = (call);                                                 \
        if (_st != synSuccess) {                                                \
            fprintf(stderr, "[FAIL] %s -> %d\n", #call, (int)_st);              \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// A graph INPUT tensor must live in an application-managed ("persistent") section, or the
// compiler's checkInputPersistence pass rejects the graph:
//   "All Inputs should be declared as persistent, User section, constant, ..."
// Give EACH input its own section -- two tensors sharing a section at the same offset
// overlap, and synLaunch rejects that.
static synSectionHandle persistent_section(synGraphHandle g) {
    synSectionHandle s = nullptr;
    if (synSectionCreate(&s, 0, g) != synSuccess) return nullptr;
    if (synSectionSetPersistent(s, true) != synSuccess) return nullptr;
    return s;
}

// Tensor sizes are listed INNERMOST-FIRST. For a row-major [rows, cols] matrix the descriptor
// is {cols, rows}. This is the most common way to get a silently-transposed GEMM -- and a
// SQUARE test cannot catch it, so the shapes below are deliberately non-square.
static synTensor make_tensor(unsigned nd, const unsigned* sizes, const char* name,
                             synSectionHandle sec) {
    unsigned s[SYN_MAX_TENSOR_DIM];
    for (unsigned i = 0; i < SYN_MAX_TENSOR_DIM; ++i) s[i] = 1;
    for (unsigned i = 0; i < nd; ++i) s[i] = sizes[i];
    synTensorDescriptor d(syn_type_float, nd, s, nullptr, false, name);
    synTensor t = nullptr;
    if (synTensorCreate(&t, &d, sec, 0) != synSuccess) return nullptr;
    return t;
}

int main(int argc, char** argv) {
    // Non-square on purpose: M=K=N would make a transposed reading indistinguishable.
    const unsigned M = 64, K = 48, N = 80;
    const int module_id = (argc > 1) ? atoi(argv[1]) : -1;   // -1 => let the runtime choose

    CHECK(synInitialize());

    synDeviceId dev = 0;
    if (module_id >= 0) {
        // Pin by PHYSICAL module id. See the manual's section on device identity: pinning by
        // a dense local index instead is a different thing and can collide silently.
        CHECK(synDeviceAcquireByModuleId(&dev, (synModuleId)module_id));
    } else {
        CHECK(synDeviceAcquireByDeviceType(&dev, synDeviceGaudi2));
    }

    synStreamHandle stream = nullptr;
    CHECK(synStreamCreateGeneric(&stream, dev, 0));

    // --- HBM allocations. Host buffers must come from synHostMalloc (a plain malloc/new
    //     pointer gives synMappingNotFound when you copy it in).
    uint64_t dA = 0, dB = 0, dC = 0;
    CHECK(synDeviceMalloc(dev, (uint64_t)M * K * sizeof(float), 0, 0, &dA));
    CHECK(synDeviceMalloc(dev, (uint64_t)K * N * sizeof(float), 0, 0, &dB));
    CHECK(synDeviceMalloc(dev, (uint64_t)M * N * sizeof(float), 0, 0, &dC));

    // --- Build the graph: one GEMM node.
    synGraphHandle graph = nullptr;
    CHECK(synGraphCreate(&graph, synDeviceGaudi2));

    // Row-major: A is [M,K] -> {K,M};  B is [K,N] -> {N,K};  C is [M,N] -> {N,M}.
    // (The "column-major" reading of this API is wrong; see the note above make_tensor.)
    unsigned sA[2] = {K, M}, sB[2] = {N, K}, sC[2] = {N, M};
    synTensor A = make_tensor(2, sA, "A", persistent_section(graph));
    synTensor B = make_tensor(2, sB, "B", persistent_section(graph));
    synTensor C = make_tensor(2, sC, "C", persistent_section(graph));
    if (!A || !B || !C) { fprintf(stderr, "[FAIL] tensor creation\n"); return 1; }

    synTensor in[2]  = {A, B};
    synTensor out[1] = {C};

    // The guid selects an OPERATION TYPE, not an engine. The graph compiler decides the
    // engine during lowering -- `gemm` happens to land on the MME, but that is the compiler's
    // node-type -> engine mapping, not the guid naming hardware. (A `matrix_multiply_dram_f32`
    // TPC kernel exists with the same intent, so "the matmul guid" is not unique.)
    synGEMMParams gemm_params(false /*transposeA*/, false /*transposeB*/);
    CHECK(synNodeCreate(graph, in, out, 2, 1,
                        &gemm_params, sizeof(gemm_params),
                        "gemm", "gemm0",
                        nullptr, nullptr));   // inputLayouts, outputLayouts
    printf("synNodeCreate(guid=\"gemm\") -> synSuccess\n");

    // --- Lower to a recipe. The 4-arg form is the only one that takes a build-log filename.
    synRecipeHandle recipe = nullptr;
    CHECK(synGraphCompile(&recipe, graph, "minimal_gemm", nullptr));
    printf("synGraphCompile -> synSuccess (recipe built)\n");

    // --- Launch. Tensors bind BY NAME here, which needs SYN_FLAGS_TENSOR_NAME.
    //     (flags = 0 would bind by tensor id, and the ids below are all 0 -- nothing binds.)
    synLaunchTensorInfo li[3];
    memset(li, 0, sizeof(li));
    li[0].tensorName = "A"; li[0].pTensorAddress = dA; li[0].tensorType = DATA_TENSOR;
    li[1].tensorName = "B"; li[1].pTensorAddress = dB; li[1].tensorType = DATA_TENSOR;
    li[2].tensorName = "C"; li[2].pTensorAddress = dC; li[2].tensorType = DATA_TENSOR;
    for (int i = 0; i < 3; ++i) {
        li[i].tensorSize[0] = (i == 0) ? K : (i == 1) ? N : N;   // innermost dim
        li[i].tensorSize[1] = (i == 0) ? M : (i == 1) ? K : M;
    }

    CHECK(synLaunch(stream, li, 3, 0, recipe, SYN_FLAGS_TENSOR_NAME));
    CHECK(synStreamSynchronize(stream));
    printf("launch + sync -> synSuccess\n");

    // The real probe does a CPU reference and compares elementwise here. The point of this
    // skeleton is the CALL SEQUENCE, not the verification.
    printf("OK -- recipe built, launched and synchronised\n");

    synRecipeDestroy(recipe);
    synGraphDestroy(graph);
    synDeviceFree(dev, dA, 0);
    synDeviceFree(dev, dB, 0);
    synDeviceFree(dev, dC, 0);
    synStreamDestroy(stream);
    synDeviceRelease(dev);
    synDestroy();
    return 0;
}
