# Third-party components

| Component | Location / use | License |
|---|---|---|
| Khronos Vulkan headers | `ray_tracing/native/include/vulkan/` | Apache-2.0; copyright and SPDX notices preserved in each header |
| NumPy | Runtime protocol and compiler arrays | BSD-3-Clause |
| Hugging Face Tokenizers | Source tokenization and incremental decoding | Apache-2.0 |
| Jinja2 | Sandboxed source chat templates | BSD-3-Clause |
| GGUF Python package | Optional GGUF metadata and coefficient reader | MIT |

The system supplies the Vulkan loader, shader compiler and C++ compiler. Local model files, datasets, executables and generated SPIR-V live outside the source distribution. Their original license files accompany local archives and model sources.

## llama.cpp Vulkan algorithms

The tiled quantized matrix, online attention reduction and clustered recurrent state kernels adapt algorithms from [llama.cpp](https://github.com/ggml-org/llama.cpp).

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
