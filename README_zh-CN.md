<div align="center">

# RecurTrace

**借助循环记忆实现自适应隐空间推理**

[论文](https://arxiv.org/abs/2609.03379) · [English](README.md) · [简体中文](README_zh-CN.md)

</div>

RecurTrace 为预训练 Qwen3 模型加入循环推理能力。共享参数的中间层通过 **Loop Memory Attention** 读取先前循环的状态，**halting head** 则为不同输入选择计算深度。一份训练好的 adapter 即可用于多种推理深度。

<p align="center">
  <img src="assets/architecture.png" width="100%" alt="RecurTrace 架构：循环记忆、共享参数的推理模块与自适应停止">
</p>

<p align="center"><em>保持预训练基座冻结，通过循环记忆和自适应深度进行推理。</em></p>

## 快速开始

使用 Python 3.10 及以上版本，推荐 Python 3.12。以下示例使用一张 CUDA GPU；添加 `--device cpu` 可在 CPU 上运行。

```bash
git clone https://github.com/JensenYX/RecurTrace.git
cd RecurTrace
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

先用仓库附带的 32 条算术与状态追踪样例训练一个小型 adapter。首次运行时会自动下载公开的 Qwen3 基座模型。

```bash
python train.py \
  --model Qwen/Qwen3-0.6B-Base \
  --data examples/train_manifest.json \
  --output outputs/stage1 \
  --max-steps 400 \
  --learning-rate 3e-4 \
  --gradient-accumulation 1 \
  --max-length 256
```

使用两次循环生成答案：

```bash
python generate.py \
  --adapter outputs/stage1 \
  --prompt "Program: x = 3; x += 4; x -= 2. What is the final value of x?" \
  --depth 2
```

命令输出包含 `answer` 和 `depth` 的 JSON：

```json
{"answer": "5", "depth": 2}
```

修改 `--depth` 即可尝试不同循环次数。示例 adapter 用于熟悉使用流程；训练面向具体任务的模型时，换成自己的任务数据即可。

`outputs/stage1` 保存 adapter 权重、配置和 tokenizer，冻结的基座单独加载。`--model` 也接受本地基座目录；使用本地基座训练后，生成时传入相同目录。

## 使用自己的数据训练

训练数据采用 JSONL，每行一段对话，最后一条 assistant 消息是需要学习的答案：

```json
{"messages":[{"role":"system","content":"Answer with the final answer only."},{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

可以将文件直接传给 `--data`，也可以用 JSON manifest 组合多个文件。路径相对于 manifest 所在目录，`weight` 控制每个训练轮次中各文件的采样量。

```json
{"files":[{"path":"math.jsonl","weight":1.0},{"path":"state_tracking.jsonl","weight":1.0}]}
```

默认训练会随机采样至多八次循环，仅优化循环记忆和输入重注入模块。较长的样例可以增大 `--max-length`。通过 `--epochs`、`--max-steps`、`--learning-rate`、`--batch-size` 和 `--gradient-accumulation` 调整训练预算。

| 公开基座模型 | 循环起始层 | 循环层数 |
| :--- | :---: | :---: |
| `Qwen/Qwen3-0.6B-Base` | 12 | 3 |
| `Qwen/Qwen3-1.7B-Base` | 12 | 3 |
| `Qwen/Qwen3-4B-Base` | 15 | 3 |
| `Qwen/Qwen3-8B-Base` | 15 | 3 |

程序会根据模型配置自动选择循环层。层索引从零开始，也可以通过 `--loop-start` 和 `--loop-layers` 指定其他位置。

多卡训练使用 `torchrun` 启动同一个脚本。下面用两张 GPU 运行仓库附带的数据：

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --model Qwen/Qwen3-0.6B-Base \
  --data examples/train_manifest.json \
  --output outputs/distributed \
  --max-steps 10 \
  --gradient-accumulation 1 \
  --max-length 256
```

## 自适应深度

完成循环 adapter 训练后，可以继续训练 halting head。首先收集不同深度下的答案损失及对应的 prompt 状态：

```bash
python prepare_oracle.py \
  --adapter outputs/stage1 \
  --data examples/halting.jsonl \
  --output outputs/oracle \
  --max-length 256
```

然后训练 halting head：

```bash
python train_halting.py \
  --adapter outputs/stage1 \
  --oracle outputs/oracle \
  --output outputs/stage2
```

根据输入自动选择深度并生成：

```bash
python generate.py \
  --adapter outputs/stage2 \
  --prompt "Program: x = 3; x += 4; x -= 2. What is the final value of x?" \
  --adaptive
```

输出包含所选深度和继续循环的概率。默认参数为 `--min-depth 2`、`--max-depth 16`、`--threshold 0.5`；降低阈值会允许更多循环。用于自己的任务时，在训练数据上拟合 head，在验证数据上选择阈值。生成整个回答时复用同一个深度。

## Python 接口

```python
import torch
from transformers import GenerationConfig
from recurtrace import load_model, load_tokenizer

adapter = "outputs/stage1"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = load_tokenizer(adapter=adapter)
model = load_model(adapter=adapter, device=device)

messages = [
    {"role": "system", "content": "Answer with the final answer only."},
    {"role": "user", "content": "What is 2 + 2?"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
config = GenerationConfig(
    max_new_tokens=32, do_sample=False, use_cache=False,
    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
)
with torch.inference_mode():
    output = model.generate(**inputs, loop_count=2, generation_config=config)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

加载 Stage 2 adapter 后，`model.choose_depth(**inputs)` 返回所选深度和继续循环概率。将该深度作为 `loop_count` 传入 `generate` 即可。

## 论文结果

在 1.7B 基座的 MathQA 实验中，RecurTrace 以平均 **2.04 次循环**达到 **56.9% 准确率**，比最佳固定深度高 **2.2 个百分点**。

<p align="center">
  <img src="assets/mathqa.png" width="700" alt="RecurTrace 与其他自适应深度方法的 MathQA 准确率和平均循环次数">
</p>

在 22 个任务上，固定两次循环的生成准确率在四个模型规模上均超过相同训练预算的微调基线：

| 模型规模 | 基线 | RecurTrace | 提升 |
| :---: | ---: | ---: | ---: |
| 0.6B | 35.2% | **35.8%** | +0.6 个百分点 |
| 1.7B | 44.8% | **46.0%** | +1.2 个百分点 |
| 4B | 58.6% | **60.8%** | +2.2 个百分点 |
| 8B | 66.6% | **70.0%** | +3.4 个百分点 |

图片与结果均来自[论文](https://arxiv.org/abs/2609.03379)。

## 引用

```bibtex
@article{wang2026recurtrace,
  title={RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory},
  author={Wang, Yuxiang and Feng, Kunyu and Shen, Yingda and Xu, Haoning and Wang, Junyu and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2609.03379},
  year={2026}
}
```

## 许可证

[Apache-2.0](LICENSE)。基于 [Qwen3](https://github.com/QwenLM/Qwen3) 和 [Hugging Face Transformers](https://github.com/huggingface/transformers) 构建。
