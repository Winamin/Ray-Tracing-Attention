# Ray Tracing Attention

[English](README.md) | [简体中文](README.zh-CN.md)

![rt_geometry 推理流程](docs/images/ray_tracing_attention_workflow.png)

**由射线求交返回数值的模型推理：每一次加法、乘法、除法、指数和取最大值都来自硬件 ray query 的命中距离；着色器只负责编码操作数、提交查询、汇合返回值和同步。**

Ray Tracing Attention（RTA）是面向大语言模型的 Vulkan 推理栈。模型通过统一编译器转译，GGUF 文件与 Hugging Face safetensors 目录共用一个入口；原生 worker 直接使用源权重，不需要训练，也不展开第二份浮点模型，Q4_K、Q6_K、BF16、FP16 与 F32 保持原始打包。选择 `rt_geometry` 后端时，完整注意力、SwiGLU MLP 与 LM Head 读出通过常驻的算术 BVH 求交完成，不再由着色器执行浮点指令。零训练转译与原生生成已在公开的 QwenFlash 9B 与 Qwen3.5 2B 上、以默认后端验证。

```powershell
rt-chat --attention-backend rt_geometry --mlp-backend rt_geometry
```

默认后端仍是 `standard`。Attention 与 MLP 可以分别选择，选择任意一个也会启用几何 LM Head。

## 算术如何进入几何

一次查询就是一个完整的运算。着色器用实例掩码选择运算场，设置射线起点与方向，再读取已提交的命中：

$$
\mathcal Q_{\mathcal F}(\mathbf o,\mathbf d)=(t,\iota),\qquad \mathbf r(t)=\mathbf o+t\,\mathbf d .
$$

$t$ 是求交距离并承载数值，$\iota$ 是实例与图元身份并承载操作数选择。一个 TLAS 保存五个三角形场，构建一次后由所有层、矩阵行、查询头以及后续 token 复用：

| 运算场 | 几何 | 命中返回 |
|---|---|---|
| 和与商平面 | $z=y$ 上的四边形 | $t=a+b$、$t=a-b$、$t=a/b$ |
| 单位平面 | $z=1$ 上的四边形 | $t=1/b$，可复用的倒数 |
| 指数条带 | $[-1,1]$ 上 $z=2^x$ 的 4096 个四边形 | 插值得到的 $2^x$ |
| 比较面 | 两个相交四边形 | 最近命中身份选出胜出操作数 |
| 双系数平面 | 64 个模板、1024 个实例，共 65536 个系数平面 | $t=2/(1-\alpha u-\beta v)$ |

这些运算都不在着色器里求值。着色器把每个 FP32 操作数拆成符号位、二进制指数和有效数字，用整数移位对齐两个操作数，再把对齐后的幅值放到射线上；命中之后按位打包恢复指数，包括向次正规数舍入的平局取偶。对 $z=y$ 平面：

$$
\mathbf o=(0,b,-a),\ \mathbf d=(0,0,1)\ \Rightarrow\ t=a+b,
\qquad
\mathbf o=(0,a,0),\ \mathbf d=(0,0,b)\ \Rightarrow\ t=a/b .
$$

一次乘法使用两次求交：$z=1$ 返回 $1/b$，再由 $z=y$ 返回 $a/(1/b)$。减法是把另一操作数翻转符号后相加。取最大值不是比较运算：两个相交平面让向下射线先命中较小操作数，提交三角形的命中身份指出胜出的源操作数本身，因此不需要用 $4-t$ 重建它。

### 双坐标片段与完整行

权重行按两个坐标一组读取。24 位有效数字写成 256 进制，$S=c_2\,2^{16}+c_1\,2^{8}+c_0$；求值的是最高两段，即保留 16 位有效数字（相对 $1.5\times10^{-5}$），使每个双坐标点积少一条相关射线链。这次截断远低于这些模型 Q4_K / Q6_K 的量化步长。正负贡献先分通道处理，最后用几何减法求差，小残差不会丢在 $1-2/t$ 里。

模板 $p$ 是平面 $z=8p+2+(p/256)y$，实例剪切提供另一个系数以及第二个系数的高两位。输入放在方向里，实例平移不会让它发生舍入，返回距离为

$$
t=\frac{2}{1-\alpha u-\beta v}.
$$

几何除法与减法继续恢复 $\alpha u+\beta v=1-2/t$，再由二进制尺度恢复该片段的量级。整行不做截断：

$$
y_r=\sum_b h_b^{\mathsf T} w_{r,b},
$$

着色器累积片段，并通过 128 路共享树归约，每条归约边的加法仍是一次 `rtAdd()` 求交。BVH 不保存跨射线累加器，索引、缓冲和屏障都留在着色器。

![rt_geometry 双坐标求交与几何归约](docs/images/ray_tracing_attention_compact_detail.png)

源数据的位信息选择可复用的平面，返回的片段成为下一次运算的操作数，同一套标量场同时服务 softmax 归一化与最大值选择。

### 原始格式的源系数

权重不会被解包成第二份浮点副本。几何路径直接读取打包载荷并在同一空间里解码：Q4_K 的块尺度乘积与仿射最小值通过 `rtMultiply()`、`rtSubtract()` 求交，Q6_K 的尺度通过 `rtMultiply()`，BF16、FP16、F32 直接读取。历史包保持与其余运行时相同的 Q4_K / Q6_K 布局。

## 单层的执行路径

![rt_geometry 的 Attention 与 MLP 算子](docs/images/ray_tracing_attention_layer_detail.png)

对同时选择两个几何后端的完整注意力层：

1. **准备输入。** 词嵌入、RMSNorm、RoPE、Q/K 归一化与历史量化仍走源路径，交给几何路径一份准备好的状态与历史包。
2. **QKV 投影。** 一次 dispatch 完成 Q、K、输出门控与 V 的双坐标行点积，结果按通道区间写入与其他后端相同的缓冲。
3. **历史更新。** 准备好的 Q/K 写入既有的因果历史包，并更新历史 BVH。
4. **因果 QK。** 每个查询头按坐标对发射微射线，一个分数工作组覆盖 8 个历史位置，因此一条射线在这个段内复用。每个被接受的图元都要核对槽位与因果位置，同一四边形的两个三角形只保留一份回执，双坐标点积再用 `rtMultiply()` 乘缩放系数。
5. **softmax。** 最大值通过比较面与共享树选出，平移后的指数通过指数条带求值，分母通过 `rtAdd()` 树求和，概率通过 `rtDivide()` 归一化。未来位置写入零权重。
6. **值聚合。** 每个输出坐标保留独立累加器；概率准备一次倒数后供头宽的两半复用。可选输出门控在结果离开注意力缓冲前应用 `rtSigmoid()`。
7. **输出投影与残差。** 双坐标行点积，残差通过 `rtAdd()` 相加。
8. **MLP。** gate 与 up 的点积为 8 个 token 的 tile 共享同一份系数编码，`rtSwiGLU()` 形成激活，down 投影使用同一双坐标路径，残差同样回到几何运算。
9. **LM Head。** 词表行经覆盖结构核验，逐行用几何点积打分，跨工作组的比较归约选出 token。logits 相等或正负零并列时按原 token ID 顺序处理。token 由原生 worker 写回并逐 token 输出到 CLI。

### 仍在源路径上的部分

几何空间覆盖投影、QK、softmax、完整 V 聚合、注意力门控、MLP、残差相加与 LM Head。词嵌入与输入准备、RMSNorm、Q/K 归一化、RoPE、历史量化与打包、词表覆盖构建、分词，以及混合模型的递推层仍使用原有实现。

## 数值语义

指数曲面是启动时构建的 4096 段 $2^x$ 分段线性近似，FP32 三角形求交本身也有误差，最高两段有效数字又界定了系数项的量级。结果接近但不与原路径逐位一致。

下文测量使用默认后端。

## 核心能力

- **统一模型编译。** GGUF 文件与 Hugging Face safetensors 目录共用一个入口。模型规格描述词表、头维度、GQA、层顺序、RoPE 与递推状态，原生管线按该规格构建。
- **几何 Attention 与 MLP。** 光线查询因果历史 BVH、算术 BVH 与词表覆盖，每一项贡献都来自真实提交的命中。Q4_K / Q6_K 按 256 系数块并行解码，小型参数沿用源精度。
- **经过校验的回复缓存。** 已校验的回复与后继地址写入因果答案树。完整命中时复用已校验路径，直接输出缓存 token，不执行前向传播；新分支执行前向传播。
- **GPU 常驻执行。** 模型中间信号留在 GPU，原生 worker 连续生成。输入词嵌入与 LM Head 只保存一份原始载荷。
- **真正逐 token 流式。** 模型生成一个 token，CLI 立即接收并刷新。底部状态行展示 RT 查询计时、GPU 传感器、生成速度和显存；输入 `/fx` 展开详细报告。

## 开始使用

安装 Python 3.10+、支持 Vulkan ray query 的显卡驱动、Vulkan SDK 与 C++17 编译器。将 `glslangValidator`、`spirv-val` 和 `g++` 放入 PATH；Windows 使用 MinGW 工具链与 Vulkan loader。

```powershell
python -m pip install ".[compiler]"

rt-compile "path/to/model.gguf" --output .local/models/chat --activate
rt-chat
```

也可以直接编译 safetensors 模型目录：

```powershell
rt-compile "path/to/hf-model" --output .local/models/chat-hf
rt-chat --asset .local/models/chat-hf/optical_model.bin
```

源码入口为 `rt_chat.cmd`、`rt_compile.cmd`，以及 `python -m ray_tracing.runtime.cli`、`python -m ray_tracing.compiler`。

```powershell
rt-compile "path/to/model.gguf" --inspect
rt-build .local/models/chat
rt-chat --prompt "请写一个 Python 二分查找函数" --max-tokens 256
rt-chat --prompt-file task.txt --jsonl
rt-chat --capacity 1024 --profile --no-cache
rt-chat --attention-backend rt_geometry --mlp-backend rt_geometry
```

`--activate` 设置本机默认模型；`RT_MODEL` 指定模型目录，`RT_HOME` 指定本地资产与编译缓存目录。原生构建按模型规格和源码指纹复用。默认生成预算为 **65536 token**。

## 实测表现

`d8e86c2`
这是之前混合tensor的数据，当前纯RT Attention性能还在优化

| 工作负载 | 实测结果 |
|---|---:|
| 二分查找任务，128 token，普通生成 GPU | 38.48 token/s |
| 128-token 因果 Tree 重放，CLI 逐 token 生成 | 11960 token/s |
| 同一重放，包含前缀恢复的完整请求 | 10014 token/s |
| 128 次 Tree 命中，生成 GPU 合计 | 1.473 ms |
| 25-token prefill，GPU | 70.53 ms |
| 256-token prefill，GPU | 91.18 ms · 2808 token/s |
| 1024-token prefill，GPU | 375.40 ms · 2728 token/s |

四轮真实任务读取当前项目的运行时、构建器、Attention 求交与等价判重、因果缓存源码，依次分析启动与构建复用、查询粒度、完整包与列等价、缓存分支并 review。每轮核验回执进入下一轮上下文，输入由实际源码与前轮回复组成。请求耗时包含前缀处理和完整生成，词元化在请求前完成。

| 轮次 | 实际代码分析任务 | 输入词元 | 输出词元 | 首次分支请求 ms | 同分支缓存重放 ms |
|---:|---|---:|---:|---:|---:|
| 1 | 启动与构建复用 | 1058 | 115 | 5569.80 | 3.18 |
| 2 | Attention 查询粒度 | 2558 | 120 | 7526.86 | 3.60 |
| 3 | 完整包与列等价 | 4096 | 85 | 11424.91 | 3.41 |
| 4 | 因果缓存分支 review | 5792 | 121 | 20799.59 | 4.56 |

## 长上下文与显存

| 历史槽位 | RTA 历史包 MiB | BF16 KV 载荷 MiB（公式） | RTA 设备分配 MiB（cache 开/关） | 驱动新增峰值 MiB（cache 开/关） | 同源 Vulkan 新增峰值 MiB |
|---:|---:|---:|---:|---:|---:|
| 512 | 5.53 | 16.00 | 5909.94 / 5909.94 | 6341 / 6387 | 5203 |
| 1024 | 11.06 | 32.00 | 5909.94 / 5909.94 | 6438 / 6367 | 5090 |
| 2048 | 22.12 | 64.00 | 5932.06 / 5932.06 | 6473 / 6452 | 5076 |
| 4096 | 44.25 | 128.00 | 5954.19 / 5954.19 | 6442 / 6388 | 5159 |
| 8192 | 88.50 | 256.00 | 6014.44 / 6014.44 | 6554 / 6260 | — |

| 输入长度（各轮回复结束后统计） | 边界高维行总载荷 | 共享历史块与场景 |
|---:|---:|---:|
| 203 token | 49,152 bytes · 3 个边界 | 3.88 MiB |
| 472 token | 131,072 bytes · 8 个边界 | 9.32 MiB |
| 959 tokens | 229,376 bytes · 14 个边界 | 19.00 MiB |
| 1228 tokens | 311,296 bytes · 19 个边界 | 24.26 MiB |

常驻算术空间本身是五个 BLAS 与一个 TLAS，每个 worker 只构建一次；构建临时区在完成后释放，顶点、实例与加速结构保持常驻。

## 模型与算子

| 接口 | 原生配置 |
|---|---|
| 模型适配器 | Qwen3.5、Qwen3、Llama 的文本模型 |
| 系数格式 | Q4_K、Q6_K、BF16、FP16、F32，原始载荷转译 |
| 完整注意力 | GQA、偶数头维度 2-256、标准 RoPE 的部分或完整维度、可选 Q/K 归一化与输出门控 |
| 递推层 | 128 维状态头、4 阶卷积、可配置共享键头与值头；这些层保持源路径 |
| 隐藏与矩阵列宽 | 256 系数块对齐，独立隐藏、注意力与递推宽度 |
| GGUF 分词 | Qwen2 / Qwen3.5 / Llama 字节 BPE；safetensors 使用源 tokenizer 与 chat template |

新增模型通过源权重适配器与 `ModelSpec` 接入；编译器在写入前验证投影形状、系数格式和算子规格。原生 Shader 的维度、词表大小、层索引与量化管线均来自模型配置。

## CLI

CLI、Python 和原生 worker 默认开启 cache。完整命中复用已经验证的回复与状态路径，新分支执行前向传播；用 `--no-cache` 或 `cache=False` 可显式关闭。
对话默认保持简洁。`/fx` 展示本轮性能，`/rt` 展示 RT 状态，`/memory` 展示显存，`/log` 显示日志路径。`/continue N` 续写，`/retry` 重放，`/tokens N` 设置预算，`/paste` 与 `/end` 输入多行任务，`/save` 与 `/load` 保存、恢复会话。

## 源码布局

项目目录为 `Ray Tracing Attention`，其中 `ray_tracing/` 是共用模型编译器和原生运行包。安装和命令在项目根目录执行。

```text
Ray Tracing Attention/
  README.md                     英文首页：原理、使用方式与实测表现
  README.zh-CN.md               中文版 README
  pyproject.toml                安装配置与命令入口
  LICENSE / NOTICE              Apache-2.0 与版权声明
  THIRD_PARTY.md                第三方组件声明
  rt_chat.cmd / rt_compile.cmd  Windows 入口
  ray_tracing/
    compiler/                   源权重适配器、规格、转译与构建
    runtime/                    原生协议、分词、对话与实时状态
    native/src/                 Vulkan 引擎、几何场景、共享因果缓存
    native/shaders/             RT 算术场与模型 Shader
    native/include/             Vulkan 头文件
  docs/images/                  架构图 PNG
  docs/paper/images/            论文用图的 PDF
  docs/paper/                   成品论文 PDF
  .github/workflows/            源码语法与打包检查
  .local/                       本机模型、工具、构建缓存与会话
```

许可证：[Apache-2.0](LICENSE)。第三方组件的原始声明保存在 [THIRD_PARTY.md](THIRD_PARTY.md) 与 [NOTICE](NOTICE)。
