# Ray Tracing Attention

[English](README.md) | [简体中文](README.zh-CN.md)|[BuildModel](COMPILER.md)
![rt_geometry inference workflow](docs/images/ray_tracing_attention_workflow.png)

**Ray-intersection arithmetic for attention, SwiGLU, and vocabulary readout, connected to complete model states and native token output.**

Ray Tracing Attention (RTA) is a Vulkan inference stack for large language models. One compiler supports GGUF and Hugging Face safetensors. Export preserves source Q4_K, Q6_K, BF16, FP16, and F32 payloads by default, with optional FP16 conversion for selected tensors. Zero-training translation and native generation have been verified on QwenFlash 9B and Qwen3.5 2B using the default backends.

The `rt_geometry` backend queries a reusable arithmetic BVH. Shaders encode operands, recover fragments, organize reductions, and synchronize buffers. The current pure geometry path gives every coordinate pair one plane intersection, so a row costs one ray query per pair plus an FP32 reduction. Existing preparation interfaces provide normalization, positional processing, and history writes.

```powershell
rt-chat --attention-backend rt_geometry --mlp-backend rt_geometry
```

The default backend is `standard`. Attention and MLP can be selected independently; either geometric choice also enables the geometric LM Head.

## How the arithmetic moves into geometry

The ray origin, direction, and field identity encode operands. A hit returns a numerical parameter and a primitive identity:

$$
\mathcal Q_{\mathcal F}(\mathbf o,\mathbf d)=(t,\iota),\qquad
\mathbf r(t)=\mathbf o+t\mathbf d.
$$

The parameter $t$ supports value recovery; $\iota$ verifies coefficients or selects an original operand. The direction uses its submitted parameterization. One TLAS organizes five triangle fields, reused across layers, query heads, and tokens.

| Field | Geometry | Hit use |
|---|---|---|
| Sum and quotient plane | $z=y$ | $t=a+b$, $t=a-b$, $t=a/b$ |
| Unit plane | $z=1$ | prepare reusable reciprocal $1/b$ |
| Exponential strips | piecewise triangles on $[-1,1]$ | interpolate $2^x$ |
| Comparison planes | $z=x$ and $z=y$ | select an operand by nearest-hit identity |
| Paired coefficient planes | $z=z_p+c+\alpha(x-x_g)+\beta y$ | $t=c+\alpha x+\beta y$ |

For example, $\mathbf o=(0,b,-a)$ and $\mathbf d=(0,0,1)$ return $t=a+b$ on $z=y$. Starting at $(0,a,0)$ with direction $(0,0,b)$ returns $t=a/b$. Multiplication prepares $1/b$ on the unit plane, then queries the slope plane for $a/(1/b)$. Comparison uses signs and ray direction to preserve the original operand selected by the hit identity.

### Paired fragments and the full row

Rows are read in coordinate pairs. The current `rtDotPair` retains source bit encodings but queries rounded leading 8-bit coefficients. Inputs and product signs are encoded in the origin, with direction `(0,0,1)`; $x_g$ is the instance translation and $c$ the plane offset:

$$
\mathbf o=(x_g+x,y,z_p),\qquad t=c+\alpha x+\beta y.
$$

The shader restores the binary scale and removes the constant with one FP32 subtraction. Fragments then accumulate and reduce in FP32: the tree fixes the addition order, and only the coefficient products stay geometric. Under ideal intersections its root is the complete dot product of the rounded effective coefficients $\bar w$:

$$
y_r\approx\sum_b\phi_b=\bar w_r^{\mathsf T}h.
$$

![rt_geometry paired intersections and geometric reduction](docs/images/ray_tracing_attention_compact_detail.png)

Matrix cells illustrate tensor shapes, and tree edges expose the reduction dependencies. The BVH performs intersections; device buffers store accumulation state across rays.

### Pure geometry fragments

The current paired field gives every coefficient pair its own plane. 256 templates cover the second coefficient and 256 instances cover the first, so 65,536 planes span all pairs. Template $p$ is $z=8p+0.75+\frac{p}{256}y$, an instance adds the shear $\frac{g}{256}x$, and the ray starts at $(4g+x,\ y,\ 8p)$ with direction $(0,0,1)$:

$$
t=0.75+\frac{g}{256}\,x+\frac{p}{256}\,y .
$$

The coefficient keeps its sign and binary exponent; the plane identity carries the leading 8 bits of its significand, rounded to the nearest value and clamped at the top. A fragment is therefore

$$
\phi=2^{P+4}\left(t-0.75\right),
$$

where $P$ is the common binary exponent of the pair.

One traversal per coordinate pair replaces the earlier chain of radix-256 digit segments and sign channels. A row now costs one ray query per pair plus FP32 accumulation, where the first geometry version spent more than a dozen intersections per pair on digits, channels, and geometric merges.

### Source coefficients and history packets

The default path reads packed source payloads. Q4_K scales and affine minima, and Q6_K scales, use geometric multiplication and subtraction; BF16, FP16, and F32 use their format readers. Optional FP16 export converts selected quantized tensors ahead of inference: `rt-compile <model> --fp16 --fp16-scope geometry` resolves the coefficient tensors the geometry path reads element by element, and the shader then reads a pair with one `unpackHalf2x16` instead of decoding a block header per element. The scope also accepts `all`, `ffn`, `ffn_word`, and `recurrent_ffn`. The plane still uses the leading 8 bits, so F16 export changes decode cost, not fragment precision. History packets retain their layouts and are addressed through format metadata.

## One layer's path

![rt_geometry attention and MLP operators](docs/images/ray_tracing_attention_layer_detail.png)

For a full-attention layer with both geometry backends selected:

1. **Prepared input.** Embedding, RMSNorm, RoPE, Q/K normalization, and history quantization stay on the existing source path and hand over a prepared state and prepared history packets.
2. **QKV projection.** Paired row dots for Q, K, the output gate, and V in one dispatch; the result is split by channel range into the same buffers the other backends use.
3. **History update.** Prepared K/V are written into the existing causal history packets and the history BVH is updated.
4. **Causal QK.** Each query head sends one micro-ray per coordinate pair. Score workgroups process history segments and reuse each ray within its segment. Every accepted primitive is verified against its slot and causal position, the two triangles of the same quad collapse to one receipt, and the dot pair is scaled with `rtMultiply()`.
5. **Softmax.** Maximum through comparison planes and the shared tree, shifted energy through the exponential strips, denominator through the FP32 tree, probabilities through `rtDivide()`. Future positions are written as zero weight.
6. **Value aggregation.** Every output coordinate keeps its own FP32 accumulator over geometric products; the probability is prepared once as a reciprocal and reused for both halves of the head width. An optional output gate applies `rtSigmoid()` before the result leaves the attention buffer.
7. **Output projection and residual.** Paired row dots, then the residual add through `rtAdd()`.
8. **MLP.** Gate and up run as two passes that share the input and keep their own coefficient encodings within a token tile; the up pass applies `rtSwiGLU()` against the stored gate value, the down projection uses the same paired path, and the residual add goes back through `rtAdd()`.
9. **LM Head.** Vocabulary rows are verified against the covering structure, each row scores with geometric dots, and a comparison reduction across workgroups selects the token. Equal logits and signed zeros keep the original token-ID order. The token is written by the native worker and streamed to the CLI.

### What still runs on the source path

The geometry space covers projections, QK, softmax, full V aggregation, the attention gate, the MLP, the residual adds, and the LM Head. Embeddings and input preparation, RMSNorm, Q/K normalization, RoPE, history quantization and packing, the vocabulary covering build, tokenization, and the recurrent layers of hybrid models still use their existing implementations.

## Numerics

The current path includes 8-bit coefficient rounding, FP32 intersection error, and exponential-strip interpolation. Complete coordinate coverage and numerical equivalence require separate validation. Fragment recovery also includes scale restoration and offset subtraction error.

Dot fragments, the exponential, maximum, division, sigmoid, and residual adds remain geometric; partial sums and reduction trees accumulate in FP32 in the shader.

The tables below were measured with the default backends.

## Key capabilities

- **Unified model compilation.** GGUF files and Hugging Face safetensors directories share one entry point. The model spec describes the vocabulary, head dimensions, GQA, layer order, RoPE, and recurrent state, and the native pipeline is built from that spec.
- **Geometric attention and MLP.** Rays query the causal history BVH, the arithmetic BVH, and the vocabulary covering; every contribution comes from a real committed hit. Q4_K / Q6_K blocks decode 256 coefficients in parallel, and small parameters keep their source precision.
- **Validated reply cache.** Each verified reply and its successor address enter the causal answer tree. A full request hit replays the verified path and streams cached tokens without a forward pass; a new branch runs the forward pass.
- **GPU-resident execution.** Intermediate signals stay on the GPU, and the native worker generates continuously. Input embeddings and the LM Head keep a single copy of the original payload.
- **True per-token streaming.** The CLI receives and refreshes each token as the model emits it. The status line shows RT query timing, sensor readings, generation speed, and GPU memory; `/fx` expands the detailed report.

## Getting started

Install Python 3.10+, a GPU driver with Vulkan ray query support, the Vulkan SDK, and a C++17 compiler. Put `glslangValidator`, `spirv-val`, and `g++` on the PATH; on Windows use the MinGW toolchain with the Vulkan loader.

```powershell
python -m pip install ".[compiler]"

rt-compile "path/to/model.gguf" --output .local/models/chat --activate
rt-chat
```

A safetensors model directory can be compiled directly:

```powershell
rt-compile "path/to/hf-model" --output .local/models/chat-hf
rt-chat --asset .local/models/chat-hf/optical_model.bin
```

Source entry points are `rt_chat.cmd`, `rt_compile.cmd`, `python -m ray_tracing.runtime.cli`, and `python -m ray_tracing.compiler`.

```powershell
rt-compile "path/to/model.gguf" --inspect
rt-build .local/models/chat
rt-compile "path/to/model.gguf" --output .local/models/chat-f16 --fp16 --fp16-scope geometry
rt-chat --prompt "write a Python binary search function" --max-tokens 256
rt-chat --prompt-file task.txt --jsonl
rt-chat --capacity 1024 --profile --no-cache
rt-chat --attention-backend rt_geometry --mlp-backend rt_geometry
```

`--activate` sets the machine-default model; `RT_MODEL` points to a model directory and `RT_HOME` to the local asset and build-cache directory. Native builds are reused by model spec and source fingerprint. The default generation budget is **65536 tokens**.

## Measured results

`d8e86c2`. This is the data from the previous mixed tensor setup. The performance of the current pure RT Attention is still under optimization.

| Workload | Measured result |
|---|---:|
| Binary-search task, 128 tokens, plain generation GPU | 38.48 token/s |
| 128-token causal Tree replay, CLI per-token generation | 11960 token/s |
| Same replay as a full request including prefix recovery | 10014 token/s |
| 128 Tree hits, generation GPU total | 1.473 ms |
| 25-token prefill, GPU | 70.53 ms |
| 256-token prefill, GPU | 91.18 ms · 2808 token/s |
| 1024-token prefill, GPU | 375.40 ms · 2728 token/s |

Four rounds of real tasks read this project's runtime, builder, attention intersection and exact duplicate classification, and causal cache sources, analyzing startup and build reuse, query granularity, full packets and column equivalence, and cache branches in turn, with a review at the end. Each round's verification receipts enter the next round's context, inputs come from the actual sources and prior replies, request time includes prefix processing and full generation, and tokenization finishes before the request.

| Round | Code analysis task | Input tokens | Output tokens | First branch request ms | Same-branch cache replay ms |
|---:|---|---:|---:|---:|---:|
| 1 | Startup and build reuse | 1058 | 115 | 5569.80 | 3.18 |
| 2 | Attention query granularity | 2558 | 120 | 7526.86 | 3.60 |
| 3 | Full packets and column equivalence | 4096 | 85 | 11424.91 | 3.41 |
| 4 | Causal cache branch review | 5792 | 121 | 20799.59 | 4.56 |

## Long context and memory

| History slots | RTA history packet MiB | BF16 KV payload MiB (formula) | RTA device allocation MiB (cache on/off) | Driver peak increment MiB (cache on/off) | Same-source Vulkan peak increment MiB |
|---:|---:|---:|---:|---:|---:|
| 512 | 5.53 | 16.00 | 5909.94 / 5909.94 | 6341 / 6387 | 5203 |
| 1024 | 11.06 | 32.00 | 5909.94 / 5909.94 | 6438 / 6367 | 5090 |
| 2048 | 22.12 | 64.00 | 5932.06 / 5932.06 | 6473 / 6452 | 5076 |
| 4096 | 44.25 | 128.00 | 5954.19 / 5954.19 | 6442 / 6388 | 5159 |
| 8192 | 88.50 | 256.00 | 6014.44 / 6014.44 | 6554 / 6260 | — |

| Input length (counted after each round's reply) | Boundary high-dimensional row payload | Shared history blocks and scene |
|---:|---:|---:|
| 203 tokens | 49,152 bytes · 3 boundaries | 3.88 MiB |
| 472 tokens | 131,072 bytes · 8 boundaries | 9.32 MiB |
| 959 tokens | 229,376 bytes · 14 boundaries | 19.00 MiB |
| 1228 tokens | 311,296 bytes · 19 boundaries | 24.26 MiB |

The resident arithmetic space itself is five BLAS and one TLAS built once per worker; its build scratch is released after construction and the vertices, instances, and backing storage stay resident.

## Models and operators

| Interface | Native configuration |
|---|---|
| Model adapters | Text models of Qwen3.5, Qwen3, and Llama |
| Coefficient formats | Q4_K, Q6_K, BF16, FP16, F32, original payloads translated |
| Full attention | GQA, even head dimensions 2-256, partial or full standard RoPE dimensions, optional Q/K normalization and output gating |
| Recurrent layers | 128-dim state heads, 4th-order convolution, configurable shared key and value heads; these layers keep the source path |
| Hidden and matrix widths | 256-coefficient block alignment; independent hidden, attention, and recurrence widths |
| GGUF tokenization | Qwen2 / Qwen3.5 / Llama byte BPE; safetensors uses the source tokenizer and chat template |

New models plug in through source-weight adapters and the `ModelSpec` interface; the compiler validates projection shapes, coefficient formats, and operator specs before writing. The native shaders take dimensions, vocabulary size, layer indices, and the quantization pipeline from the model configuration.

## CLI

The CLI, Python, and the native worker enable cache by default. A full hit reuses the verified reply and state path and a new branch runs the forward pass; use `--no-cache` or `cache=False` to disable it explicitly.
The conversation stays compact by default. `/fx` shows per-round performance, `/rt` shows RT status, `/memory` shows GPU memory, and `/log` shows the log path. `/continue N` continues writing, `/retry` replays, `/tokens N` sets the budget, `/paste` and `/end` handle multi-line input, and `/save` and `/load` save and restore sessions.

## Repository layout

The project directory is `Ray Tracing Attention`, where `ray_tracing/` holds the shared model compiler and native runtime package. Install and run commands from the project root.

```text
Ray Tracing Attention/
  README.md                     Principles, usage, and measured results (English)
  README.zh-CN.md               Chinese README
  pyproject.toml                Install configuration and command entry points
  LICENSE / NOTICE              Apache-2.0 and copyright notices
  THIRD_PARTY.md                Third-party component declarations
  rt_chat.cmd / rt_compile.cmd  Windows entry points
  ray_tracing/
    compiler/                   Source-weight adapters, specs, translation, and builds
    runtime/                    Native protocol, tokenization, chat, and live status
    native/src/                 Vulkan engine, geometry scenes, shared causal cache
    native/shaders/             RT arithmetic fields and model shaders
    native/include/             Vulkan headers
  docs/images/                  Architecture figures as PNG
  docs/paper/images/            Paper figures as PDF
  docs/paper/                   Finished paper PDF
  .github/workflows/            Source syntax and packaging checks
  .local/                       Local models, tools, build caches, and sessions
```

License: [Apache-2.0](LICENSE). Original declarations of third-party components are kept in [THIRD_PARTY.md](THIRD_PARTY.md) and [NOTICE](NOTICE).
