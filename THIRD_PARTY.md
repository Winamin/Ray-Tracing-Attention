# Third-party components

| Component | Location / use | License |
|---|---|---|
| Khronos Vulkan headers | `pure_rt_attention/native/include/vulkan/` | Apache-2.0; copyright and SPDX notices preserved in each header |
| NumPy | Runtime protocol and compiler arrays | BSD-3-Clause |
| Hugging Face Tokenizers | Source tokenization and incremental decoding | Apache-2.0 |
| Jinja2 | Sandboxed source chat templates | BSD-3-Clause |
| GGUF Python package | Optional GGUF metadata and coefficient reader | MIT |

The system supplies the Vulkan loader, shader compiler and C++ compiler. Local model files, datasets, executables and generated SPIR-V live outside the source distribution. Their original license files accompany local archives and model sources.
