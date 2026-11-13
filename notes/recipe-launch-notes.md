# How a recipe launches

**Status:** working notes for our own team, not a manual chapter. Written 2026-11-13 from one
measured experiment (`experiments/`, rungs 0–1) plus a read of the installed Synapse headers and the
bridge source at the exact version installed here.

**Provenance.**
- Bridge source: `HabanaAI/gaudi-pytorch-bridge` @ `27eccbc` ("Publish v1.24.2"), cloned read-only.
  Paths below are relative to that tree.
- Synapse API: `/usr/include/habanalabs/synapse_api.h` and `synapse_api_types.h` as installed.
- Runtime: driver `1.24.2-52a555d`, torch `2.13.0a0+git2469d18`, `habana-torch-plugin 1.24.2`.
- Experiment: `experiments/01_one_region.py` (run it with `experiments/run.sh`, which sets the cache
  contract and reports counts).

**Basis tags.** **MEASURED** (ran here, artefact named) · **source read** (a named file and line) ·
**INFERRED** (derivation stated) · **UNKNOWN**.

---

## 0. The experiment that anchors this

One `torch.compile` region over a fixed-shape 1-D tensor:

```python
N = 1024
f = torch.compile(lambda t: t + 1, backend="hpu_backend")
y = f(torch.arange(N, dtype=torch.float32, device="hpu"))
```

| observation | value | basis |
|---|---|---|
| dynamo | `graph_count=1, break_count=0, op_count=1` | MEASURED |
| FX graph | `placeholder → operator.add(%t, 1) → return` | MEASURED |
| **recipes produced** | **1** | MEASURED |
| recipe file | `4169968267915495333_a16fb581eee_syn1.24.2.79e284b.recipe` | MEASURED |
| companion files | `.metadata`, `.recipe_debug_files/` | MEASURED |
| first call | 140.5 ms cold, 76.7 ms with the recipe on disk | MEASURED |
| second call, same process | 0.5–0.7 ms | MEASURED |
| same function, **1 M elements** (4 MiB), first call | 103.6 ms — **same as 4 KiB** | MEASURED |
| three functions, identical bodies, different names | **one** recipe | MEASURED |

So: one region → one recipe per shape; the compile cost is independent of data size; and the Python
function name is **not** part of the recipe identity.

---

## 1. What Synapse exposes

All declarations in `/usr/include/habanalabs/synapse_api.h`. The ones that matter for recipes:

| purpose | call | line |
|---|---|---|
| build a graph | `synGraphCreate`, `synTensorCreate`, `synNodeCreate`, `synNodeDependencySet` | — |
| **compile** | `synGraphCompile(&recipe, graph, recipeName, buildLog)` | 1378 |
| compile with error handle | `synGraphCompileV2(&recipe, graph, name, &errorHandle)` | 1397 |
| **write a recipe to disk** | `synRecipeSerialize(recipe, fileName)` | 292 |
| **read a recipe back** | `synRecipeDeSerialize(&recipe, fileName)` | 305 |
| query sizes | `synRecipeGetAttribute(&retVal, attrs, n, recipe)` | 323 |
| query section layout | `synRecipeSectionGetProp(recipe, section, prop, &v)` | 445 |
| free | `synRecipeDestroy(recipe)` | 279 |
| **workspace size** | `synWorkspaceGetSize(&size, recipe)` | 641 |
| **launch** | `synLaunch(stream, infos, n, workspace, recipe, flags)` | 533 |
| launch with events | `synLaunchWithExternalEvents(...)` | (used by the bridge, §3) |
| launch metadata | `synTensorRetrieveLaunchAmount` / `…LaunchIds` / `…LaunchInfoById` | 928 / 947 / 977 |

**`synRecipeAttribute` carries sizes only** (`synapse_api_types.h:37-45`): workspace size, number and
size of persistent tensors, host-memory size, number of external tensors, const-section size, device
memory size. **There is no attribute for a recipe id, hash, or cache key** — the identity a recipe has
is the *file name* it was serialised under.

**There is no cache API at all.** `grep -i cache synapse_api.h` returns nothing. Caching is a property
of the layers above, not of Synapse's public interface.

---

## 2. What the bridge does with it — the compile side

### 2.1 The name is a compile argument, and it is per-process by construction

`backend/synapse_helpers/graph.cpp:643`:

```cpp
auto name = get_unique_recipe_name(name_, eager_mode_);
status = synGraphCompile(&recipe_handle->syn_recipe_handle_, graph_handle_, name.c_str(), nullptr);
...
recipe_handle->recipe_name_ = std::move(name);
```

`graph.cpp:71`:

```cpp
std::string get_unique_recipe_name(const std::string& name, bool eager_mode) {
  static uint64_t suffix = 0;
  static std::string graph_dump_dir = check_and_prepare_graph_dump_dir();
  if (eager_mode && !(IS_SYNHELPER_DEBUG_ENABLED)) return graph_dump_dir + std::to_string(suffix++);
  return absl::StrFormat("%s%s_%d", graph_dump_dir, name, ++suffix);
}
```

So the recipe **name** is `<graph_dump_dir><graph name>_<counter>` — a static counter, unique per
process, and it embeds the (CWD-relative) dump directory. That is why post-graph dumps are named
`.graph_dumps/graph_0000_fused_0_syn_…_2` (**source read**).

**The name is metadata, not identity.** It appears in the dumps, in the logs, and in the serialised
metadata — and in no cache key (§2.2).

### 2.2 The bridge has its own disk cache, keyed on a hash it computes

`backend/kernel/hpu_recipe_cache.cpp:100`:

```cpp
std::shared_ptr<RecipeHolder> RecipeCacheLRU::get(std::shared_ptr<RecipeArgumentSpec>& key) {
  if (exists(key)) { ...; return list_.front().second; }     // in-memory LRU hit
  else if (disk_cache_) {
    auto val = disk_cache_->Find(*key);                       // disk lookup
    if (val) { insert(key, val); return val; }
  }
  return {nullptr};                                           // miss: caller compiles
}
```

`:350`:

```cpp
std::shared_ptr<RecipeHolder> DiskCache::Find(const RecipeArgumentSpec& spec) {
  std::stringstream ss;
  auto recipe = recipe_cache_.lookup(std::to_string(spec.hashCode()) + cache_id_suffix_, ss);
  return recipe ? std::make_shared<RecipeHolder>(ss, *recipe) : nullptr;
}
```

`:294` — the suffix that completes the key:

```cpp
if (GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_IGNORE_VERSION)) cache_id_suffix_ = "";
else {
  synDriverGetVersion(versionStr.data(), bufferSize);
  cache_id_suffix_ = absl::StrCat("_", CacheVersion::libs_env_hash(), "_syn", versionStr.data());
}
```

So the on-disk identity is:

```
<RecipeArgumentSpec::hashCode()> _ <libs_env_hash> _ syn<driver version> .recipe
4169968267915495333              a16fb581eee        1.24.2.79e284b
```

### 2.3 What is in the spec hash — and the one term that makes it fragile

`backend/kernel/hpu_habana_cache.cpp:216-240`:

```cpp
hash_code = hash_combine(
  hash_combine(
    hash_combine(
      hash_combine(
        hash_combine_scalars(                       // scalar input *values*
          hash_combine(hash_combine(cas.hashCode(), graph_hash_code),
                       irgraph->outputs().size()),
          input_refs),
        offset_hash_code),                          // device pointer offsets
      h2d_hash_code),                               // host-to-device bucketing data
    symhash),
  permhash);
```

and `:25-40`:

```cpp
auto storage_data_ptr_ = reinterpret_cast<synapse_device_ptr>(pt_tensor.storage().data_ptr().get());
auto buffer_ptr       = reinterpret_cast<synapse_device_ptr>(pt_tensor.data_ptr());
auto offset = (buffer_ptr - storage_data_ptr_);
offset_hash_code = at::hash_combine(offset_hash_code, offset);
```

**The recipe key includes the device-pointer offset of every input tensor within its storage.** Not
the address — the offset — but any change in allocation layout that shifts a tensor inside its storage
changes the key. A same-configuration reboot repeats the layout and hits; a different bucket set or
pool size shifts it and misses (**INFERRED** from the code plus the measured 0-vs-2,168 recipe counts
across two configurations).

### 2.4 What the version component hashes

`habana_serialization/cache_version.cpp:190-220` — `libs_env_hash()` combines:

1. a **whole-file hash of `libhabana_pytorch_backend.so`** (itself, via `dladdr`),
2. a **whole-file hash of `libSynapse.so`** (located through `/proc/self/maps`),
3. the **content of every file in `GC_KERNEL_PATH`** (it tries to split a list),
4. **16 named environment variables** that affect compilation — `ENABLE_RAGGED_SOFTMAX_OPT`,
   `TPC_ENGINES_ENABLED_MASK`, `DISABLE_SYNAPSE_QUANTIZATION`, `PROFILE_PRECISION`,
   `MME_STRATEGY_ALIGNED_ADDRESSES_ENABLED`, and 11 more; the comment says the list was taken from
   `habana_global_conf.cpp | grep MakePublic` (**source read**).

So changing a library, a kernel library, or one of those variables invalidates the entire cache —
a well-defined contract, and a much stronger one than the spec hash's offsets.
`PT_RECIPE_CACHE_IGNORE_VERSION` removes this component entirely.

### 2.5 Store and load are `synRecipeSerialize` / `synRecipeDeSerialize` — the bridge calls them itself

`habana_serialization/recipe_cache.cpp:36-70` (load):

```cpp
if (file_exists(recipe_path)) {
  synRecipeHandle recipeHandle;
  auto status = synRecipeDeSerialize(&recipeHandle, recipe_path.c_str());   // :58
  ...
  return recipeHandle;
}
return nullptr;   // "probably empty recipe cached"
```

`:163` (store, inside `lockfree_store_task`):

```cpp
auto serialize_status = synRecipeSerialize(recipeHandle->syn_recipe_handle_, recipe_path.c_str());
// then the metadata stream is written to <key>.metadata
```

File naming (`cache_file_handler.cpp:40`):

```cpp
std::string recipe_file_path(std::string const& path, const std::string& cache_id) {
  return path + "/" + cache_id + RECIPE_SUFFIX;
}
```

---

## 3. What running a recipe looks like

### 3.1 Once, after compile or deserialize: interrogate the recipe

```cpp
synWorkspaceGetSize(&workspace_size, recipe_handle.syn_recipe_handle_);            // graph.cpp:720
synTensorRetrieveLaunchAmount(recipe.syn_recipe_handle_, &numOfTensors);           // lowering:602
synTensorRetrieveLaunchIds(recipeHandle, ids.data(), numOfTensors);                // lowering:315
synTensorRetrieveLaunchInfoById(recipeHandle, numOfTensors, tensorInfos.data());   // lowering:321
```

That yields the workspace size and a per-tensor launch-info template — name, size, dtype, dims. The
bridge keeps it as the "patching table" for the recipe (**source read**).

### 3.2 Per step: patch addresses, then one launch

`graph.cpp:761-950` (`graph::launch`) fills a `std::vector<synLaunchTensorInfo>` with the **current**
device addresses and calls:

```cpp
if (!GET_ENV_FLAG_NEW(PT_COMPILE_ONLY_MODE)) {
  status = synLaunchWithExternalEvents(
      compute_stream,                       // synStreamHandle
      inputs_and_outputs_info.data(),       // synLaunchTensorInfo* (name, address, size)
      static_cast<uint32_t>(inputs_and_outputs_info.size()),
      workspace_buffer,                     // uint64_t
      recipe_handle.syn_recipe_handle_,     // the recipe
      event_handles.data(),                 // external events to wait on
      static_cast<uint32_t>(event_handles.size()),
      flags);
}
```

Around it: `towl::emitRecipeLaunch` twice (logical then physical addresses), device-memory stats, and
`HABANA_ASSERT(status == synSuccess)`.

```
compile or deserialize  →  recipe handle + workspace size + launch-info template   (once)
patch addresses         →  synLaunchTensorInfo[] with current device pointers     (per step)
synLaunchWithExternalEvents(stream, infos, n, workspace, recipe, events, flags)   (per step)
optional sync + trace
```

### 3.3 Two knobs worth knowing

| knob | effect |
|---|---|
| `PT_COMPILE_ONLY_MODE` | skips the launch entirely — isolates compile cost from execution cost in one process |
| `PT_HPU_SYNC_LAUNCH=1` | `device.synchronize()` after each launch — turns an async enqueue into a synchronous step |

---

## 4. The four cache layers, and what each keys on

| layer | artefact | key | lifetime | observed |
|---|---|---|---|---|
| dynamo | FX graph + guards | code object + guard values | process | ~50 ms per (function × shape) trace; **no cross-process cache** |
| **bridge** | serialised recipe + metadata | `RecipeArgumentSpec::hashCode()` + `libs_env_hash` + driver | **disk** | works; 0 new recipes on a same-config reboot |
| Synapse (internal) | a recipe's **static part**, in **device HBM** | internal `GCFG_RECIPE_CACHE_*` (`RECIPE_CACHE_SIZE` / `_BLOCK_SIZE`) — a **different cache** from `PT_HPU_RECIPE_CACHE_CONFIG` | device acquire → release | Gaudi1/qman path only; not on the Gaudi2/SCAL path at all — see [the residency note](recipe-residency-and-memory-budget.md) §4 |
| runtime | loaded recipe handle | handle in the bridge's LRU | process | 0.5–0.7 ms per launch |

**The one layer with no cross-process cache is dynamo's.** Everything below it is durable; **none of
it is content-keyed** — the key's library and environment component is dead code (the shadowed
`hashRes` in `cache_version.cpp`; see [§8.3 of the residency note](recipe-residency-and-memory-budget.md)),
so a rebuilt kernel or a changed flag is **not** detected and a stale recipe can be served. Everything
above dynamo is process-local.

---

## 5. Traps found while doing this

1. **`PT_HPU_GRAPH_DUMP_MODE=0` is rejected** — `Value '0' is not allowed. Allowed values: compile |
   eager | all` — and kills the process with a FATAL ERROR. Leave it unset unless you want graphviz
   dumps (**MEASURED**).
2. **Two dump mechanisms.** `DUMP_POST_GRAPHS=<existing dir>` writes one JSON per compiled graph into
   `<dir>/.graph_dumps/`; a second, graphviz-style mechanism writes `.graph_dumps/` **relative to the
   CWD**. Both appeared in the same run (**MEASURED**).
3. **The inductor cache ignores `TMPDIR`** — it writes `/tmp/torchinductor_$USER` unless
   `TORCHINDUCTOR_CACHE_DIR` is set (**MEASURED**).
4. **`torch.compile` without a backend picks inductor and dies** — `InductorError: KeyError: 'hpu'`.
   The HPU backend is registered by the plugin but is not the default; name it
   (`backend="hpu_backend"`) (**MEASURED**).
5. **The recipe name embeds the CWD-relative dump directory**, so the name changes with the working
   directory — irrelevant to caching, but confusing when reading dumps (**source read**).

---

## 6. What this means for controlling the stack

- **The recipe identity is not something you pass in.** You cannot name your way into a cache hit:
  the name is metadata, and the bridge's disk key is a hash of the argument spec plus a
  library/environment hash.
- **The supported way to own the artefact is `synRecipeSerialize` / `synRecipeDeSerialize` with a
  filename you choose.** Compute your own key, write the recipe where you want it, and read it back
  before compiling. That is what `wright` does (`RecipeCache` → `{name}-{identity[:32]}.recipe`), and
  it needs no change to Synapse.
- **The offset term in the spec hash is the fragile part.** A cache that must survive allocation-layout
  changes (different bucket sets, different pool sizes, different concurrency) cannot rely on the
  bridge's key; it has to key on the graph description instead.
- **`PT_RECIPE_CACHE_IGNORE_VERSION` and `PT_COMPILE_ONLY_MODE`** are the two knobs that make
  experiments tractable: the first removes the library/env component from the key, the second lets a
  process compile without launching.

---

## 7. UNKNOWN

- **Whether the bridge or Synapse writes the `.recipe` file.** The bridge has the `CacheFileHandler`
  and the "Storing in disc cache" message and calls `synRecipeSerialize` with a path — but
  `synRecipeSerialize` itself writes, so the file is Synapse's format either way. Which layer does the
  `open()` is **UNKNOWN** (both are consistent with the evidence).
- **What `CacheVersion::libs_env_hash()`'s `GC_KERNEL_PATH` splitter actually does with a
  colon-separated list** — the code looks for `:` first and then splits on `,`; the loop structure
  suggests an inconsistency, but its effect on the hash is **UNKNOWN** without a test.
- **Whether `PT_HPU_ENABLE_CACHE_METRICS=1` prints hit/miss counters** — the strings exist
  (`#original_recipe_hits=`, `#refined_recipe_hits=`), but nothing was printed in our run. **UNKNOWN**
  whether they need a log level or appear at exit.
- **The exact set of constructors actually used** by the vLLM/torch.compile path (there are five
  `RecipeArgumentSpec` constructors with different key compositions). Our measurements are consistent
  with the `irgraph` constructor, which includes offsets and h2d data, but that is **INFERRED**.
