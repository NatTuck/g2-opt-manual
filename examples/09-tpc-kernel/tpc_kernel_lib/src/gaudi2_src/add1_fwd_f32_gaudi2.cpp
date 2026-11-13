/**********************************************************************
  Glue code for add1_fwd_f32_gaudi2 TPC kernel.

  Pattern: src/Habana_Custom_Kernel/src/gaudi2_src/add_f32_gaudi2.cpp
  - Uses elementsInVec=64 for FP32
  - Returns GLUE_INSUFFICIENT_ELF_BUFFER when buffer too small
 **********************************************************************/

#include <vector>
#include <cstring>
#include "add1_fwd_f32_gaudi2.hpp"

extern unsigned char _binary___add1_fwd_f32_gaudi2_o_start;
extern unsigned char _binary___add1_fwd_f32_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn Add1FwdF32Gaudi2::GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, "add1_fwd_f32_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn Add1FwdF32Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    const int c_unrollCount = 4;

    if (in_defs->inputTensorNr != 1)
    {
        in_defs->inputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // Shapes must match
    if (in_defs->inputTensors[0].geometry.maxSizes[0] != in_defs->outputTensors[0].geometry.maxSizes[0] ||
        in_defs->inputTensors[0].geometry.maxSizes[1] != in_defs->outputTensors[0].geometry.maxSizes[1] ||
        in_defs->inputTensors[0].geometry.maxSizes[2] != in_defs->outputTensors[0].geometry.maxSizes[2] ||
        in_defs->inputTensors[0].geometry.maxSizes[3] != in_defs->outputTensors[0].geometry.maxSizes[3])
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    // FP32 only for now
    if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
        in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    // elementsInVec=64 for FP32
    int elementsInVec = 64;

    uint64_t inputSizes[gcapi::MAX_TENSOR_DIM] = {0};
    memcpy(inputSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(inputSizes));

    unsigned depthIndex = (inputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    if (depthIndex == 0) depthIndex = 1;

    out_defs->indexSpaceRank = 5;
    out_defs->indexSpaceGeometry[0] = depthIndex;
    out_defs->indexSpaceGeometry[1] = (inputSizes[1] + (c_unrollCount - 1)) / c_unrollCount;
    out_defs->indexSpaceGeometry[2] = inputSizes[2];
    out_defs->indexSpaceGeometry[3] = inputSizes[3];
    out_defs->indexSpaceGeometry[4] = 1;

    // Input access pattern
    out_defs->inputTensorAccessPattern[0].allRequired = true;
    out_defs->inputTensorAccessPattern[0].mapping[0].indexSpaceDim = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].a = elementsInVec;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b = elementsInVec - 1;

    out_defs->inputTensorAccessPattern[0].mapping[1].indexSpaceDim = 1;
    out_defs->inputTensorAccessPattern[0].mapping[1].a = c_unrollCount;
    out_defs->inputTensorAccessPattern[0].mapping[1].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[1].end_b = c_unrollCount - 1;

    for (unsigned int dims = 2; dims < out_defs->indexSpaceRank; dims++) {
        out_defs->inputTensorAccessPattern[0].mapping[dims].indexSpaceDim = dims;
        out_defs->inputTensorAccessPattern[0].mapping[dims].a = 1;
        out_defs->inputTensorAccessPattern[0].mapping[dims].start_b = 0;
        out_defs->inputTensorAccessPattern[0].mapping[dims].end_b = 0;
    }

    // Output access pattern
    out_defs->outputTensorAccessPattern[0].allRequired = true;
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a = elementsInVec;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b = elementsInVec - 1;

    out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].a = c_unrollCount;
    out_defs->outputTensorAccessPattern[0].mapping[1].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[1].end_b = c_unrollCount - 1;

    for (unsigned int dims = 2; dims < out_defs->indexSpaceRank; dims++) {
        out_defs->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim = dims;
        out_defs->outputTensorAccessPattern[0].mapping[dims].a = 1;
        out_defs->outputTensorAccessPattern[0].mapping[dims].start_b = 0;
        out_defs->outputTensorAccessPattern[0].mapping[dims].end_b = 0;
    }

    // No scalar params
    out_defs->kernel.paramsNr = 0;

    // Copy kernel binary into the buffer GC provided
    unsigned IsaSize = (uint64_t)(&_binary___add1_fwd_f32_gaudi2_o_end - &_binary___add1_fwd_f32_gaudi2_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize) {
        memcpy(out_defs->kernel.kernelElf, &_binary___add1_fwd_f32_gaudi2_o_start, IsaSize);
        retVal = tpc_lib_api::GLUE_SUCCESS;
    } else {
        retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }

    return retVal;
}
