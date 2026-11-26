# Model compiler

`rt-compile` translates a source model into the resident asset directory and builds the native worker for its geometry. `rt-build` rebuilds the worker from an existing asset directory. This page lists the current compiler features; the top-level [README](README.md) covers the runtime.

```powershell
python -m pip install ".[compiler]"
rt-compile "path/to/model.gguf" --output .local/models/chat --activate
rt-chat
```

## Sources

| Source | Accepted input | Notes |
|---|---|---|
| GGUF file | `.gguf` with architectures `qwen35`, `qwen3`, or `llama` | Read through `gguf.GGUFReader`; tensors stay packed |
| Safetensors directory | `config.json` plus one or more `*.safetensors` shards | Uses the source `tokenizer.json` and chat template |

The adapters reject what the native graph cannot run:

- RoPE scaling other than `none` / `default` (`rope_scaling`, `rope_parameters`).
- Biased attention or MLP projections.
- A non-SiLU hidden activation.
- A recurrent layer whose value heads are not 128-dimensional or whose convolution is not four taps.
- Coefficient formats without a decoder.

GGUF tokenizers are translated to `tokenizer.json` when the tokenizer is byte-level BPE (`tokenizer.ggml.model = gpt2`) and its preprocessor is `qwen2`, `qwen35`, or `llama-bpe`. Other preprocessors need a safetensors source.

## Commands

```text
rt-compile SOURCE [--output DIR] [--inspect] [--no-build] [--activate]
                  [--fp16] [--fp16-scope SCOPE]
rt-build DIR [--output DIR]
```

| Option | Effect |
|---|---|
| `--output DIR` | Asset directory. Defaults to `RT_HOME/models/<source name>`. An existing `optical_model.bin` in that directory aborts the run. |
| `--inspect` | Print the model specification as JSON and exit without writing assets. |
| `--no-build` | Export the asset directory and skip the native build. |
| `--activate` | Write `active_model.json` so `rt-chat` uses the new model by default. |
| `--fp16` | Decode Q4_K / Q6_K coefficients to F16 at export time. |
| `--fp16-scope SCOPE` | Which tensors `--fp16` converts: `all` (default), `recurrent_ffn`, `ffn`, `ffn_word`, or `geometry`. |

Export runs the shape and format checks for the complete graph before it opens an output stream, then prints one progress line per layer.

`--fp16-scope geometry` converts the tensors the geometry path reads element by element: the attention projections, the gate/up/down matrices, and the LM Head. `all` additionally converts every other Q4_K / Q6_K tensor except `token_embd.weight`, which is an embedding lookup. F16 conversion changes decode cost, not the plane precision of the geometry path.

## Asset directory

| File | Contents |
|---|---|
| `optical_model.bin` | `RTRES004` header, per-layer parameter table, norms, RoPE and attention parameters, recurrent constants |
| `geometry_weights.bin` | Packed coefficient payload per layer, addressed by word offset and format code |
| `vocabulary.bin` | `RTBASQ01` basis rows shared by the embedding and the LM Head |
| `coordinate_partition.bin` | `RTPART01` selectors: 256-coordinate blocks, two-coordinate packets, and their exact inverse |
| `model.json` | Model specification, tensor payload records with a per-tensor sample hash, coordinate partition descriptor, history quantization, parameter byte count |
| `tokenizer.json` | Byte-level BPE vocabulary and merges |
| `chat_template.jinja` | Chat template from the source or its `tokenizer_config.json` |
| `generation_config.json` | `eos_token_id` and `bos_token_id` |

## Model specification

`model.json` stores a `ModelSpec`: architecture, hidden and MLP widths, vocabulary size, layer types (`full` or `recurrent`), attention and KV head counts, head dimension, rotary dimension, RoPE theta, norm epsilon, attention gate, Q/K normalization, and the recurrent state heads. Its validation rejects:

- a head dimension that is odd or larger than 256;
- a hidden width, MLP width, or attention projection width that is not a multiple of 256;
- query heads that do not divide into the KV heads;
- a recurrent layer without 128-dimensional state heads;
- a history packet wider than one 128-thread packing group;
- an odd rotary dimension, or one larger than the head dimension.

The specification fingerprint (16 hex characters) seeds the build cache.

## Build

`rt-build` compiles the native worker for one asset directory; `rt-compile` does the same unless `--no-build` is given. The build:

1. writes `rt_model_config.h` and `rt_layer_map.hpp` from the specification defines;
2. compiles the shader set for the model, adding the prefill, ordered-detector, or split-local shaders only when the specification needs them, at Vulkan 1.2 or 1.3 and with `-Os` for the split-local and prefill-decode modules;
3. validates every module with `spirv-val`;
4. links `rt_optical_resident.exe` from `native/src/optical_resident.cpp` with a C++17 compiler.

Required tools:

| Tool | Default | Override |
|---|---|---|
| Shader compiler | `RT_HOME/tools/glslang/...` when present, otherwise `glslangValidator` | `RT_GLSLANG` |
| SPIR-V validator | `spirv-val` next to the shader compiler, otherwise `spirv-val` | `RT_SPIRV_VAL` |
| C++ compiler | `g++` | `CXX` |

Builds are cached in `RT_HOME/build/<key>` and reused when the executable and its `complete` marker exist. The key covers the specification fingerprint, every file in `ray_tracing/native`, and whether the installed glslang supports vector decode.

## Environment

| Variable | Meaning |
|---|---|
| `RT_HOME` | Local data root for models, build cache, tools, and sessions. Defaults to `.local` beside the package. |
| `RT_MODEL` | Model directory that overrides the activated model. |
| `RT_GLSLANG` | Shader compiler path. |
| `RT_SPIRV_VAL` | SPIR-V validator path. |
| `CXX` | C++ compiler path. |

## Errors

| Message | Cause |
|---|---|
| `Architecture ... needs an operator adapter` | Unsupported `model_type` or GGUF architecture |
| `... coefficient format ... needs a decoder` | Unsupported quantization |
| `... expected (a, b), got (c, d)` | Tensor shape does not match the specification |
| `A tokenizer.json is required in the source directory` | Safetensors source without a tokenizer |
| `Build tool ... is required` | Missing shader compiler, validator, or C++ compiler |
| `<dir> already contains a compiled model` | `--output` points at an existing asset directory |

## Continuous integration

`.github/workflows/check.yml` byte-compiles the package, runs `rt-compile --help`, `rt-build --help`, and `rt-chat --help`, and builds the distribution wheel on every push.
