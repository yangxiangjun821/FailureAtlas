# 阶段 E 检查清单：现代视觉模型

更新日期：2026-11-04

## E1：模型代码接入

- [x] MobileNetV3-Small 已注册为 `mobilenet_v3`。
- [x] EfficientNet-B0 已注册为 `efficientnet_b0`。
- [x] ConvNeXt-Tiny 已注册为 `convnext_tiny`。
- [x] 轻量 CIFAR ViT 已注册为 `vit`（仅从头训练）。
- [x] 官方 ImageNet ViT-B/16 已注册为 `vit_b16`（仅预训练）。同一个模型 ID 只对应一种架构，权重模式之间才可比。
- [x] 五个模型均绑定 CIFAR-10、五类图像扰动和 10 类输出。
- [x] 五个模型均通过统一 `ModelAdapter` 接入，不修改智能体主流程。
- [x] 五个模型均能接受 `2×3×32×32` 输入并输出 `2×10` logits。
- [x] Dashboard 模型目录和兼容矩阵可从注册表自动读取新模型。
- [x] `baseline → scan → repair → compare` 五步流水线均能为新模型生成计划。

当前参数量：

| 模型 | 参数量 | 当前模式 |
|---|---:|---|
| MobileNetV3-Small | 1,528,106 | 从头训练 |
| EfficientNet-B0 | 4,020,358 | 从头训练 |
| ConvNeXt-Tiny | 27,827,818 | 从头训练 |
| ViT-CIFAR | 10,695,562 | 从头训练 |
| ViT-B/16 | 85,806,346 | 仅预训练 |

## E1：实际运行验收

- [ ] MobileNetV3 完成 1 epoch 冒烟训练并生成独立 `run_id`。
- [ ] EfficientNet-B0 完成 1 epoch 冒烟训练并生成独立 `run_id`。
- [ ] ConvNeXt-Tiny 完成 1 epoch 冒烟训练并生成独立 `run_id`。
- [ ] ViT-CIFAR 完成 1 epoch 冒烟训练并生成独立 `run_id`。
- [ ] 四个实验均能在 Dashboard 实验记录中打开详情。
- [ ] 检查 GPU 显存；ConvNeXt/ViT 若显存不足，将批量大小降至 32 或 16。

推荐冒烟命令：

```powershell
python -m failure_atlas.agent run --stage baseline --model mobilenet_v3 --dataset cifar10 --baseline-epochs 1 --batch-size 64 --run-id e1-mobile-smoke
python -m failure_atlas.agent run --stage baseline --model efficientnet_b0 --dataset cifar10 --baseline-epochs 1 --batch-size 64 --run-id e1-efficient-smoke
python -m failure_atlas.agent run --stage baseline --model convnext_tiny --dataset cifar10 --baseline-epochs 1 --batch-size 32 --run-id e1-convnext-smoke
python -m failure_atlas.agent run --stage baseline --model vit --dataset cifar10 --baseline-epochs 1 --batch-size 32 --run-id e1-vit-smoke
```

## E2：预训练权重

- [x] 为 MobileNetV3、EfficientNet-B0、ConvNeXt-Tiny 增加 `scratch` / `pretrained` 权重模式；ViT-B/16 仅提供 `pretrained`，紧凑 ViT 仅提供 `scratch`。
- [x] 权重模式写入产物路径：预训练产物位于 `*_pretrained` 目录，不会覆盖同一模型的从头训练基线。
- [x] 预训练微调使用 0.1 倍学习率，并把批量上限设为 64（显式 `--batch-size` 除外）；扫描和对比同样遵守批量上限。
- [x] 检查点的权重模式与运行请求不一致时直接报错，不会悄悄混用。
- [x] 目录、兼容矩阵和看板列出模型时不会下载预训练权重。
- [x] 使用 TorchVision 官方权重枚举，不使用不明来源权重。
- [x] 在基准指标和运行配置中保存权重模式、来源、枚举版本、URL、缓存位置与预处理参数。
- [x] 将 CIFAR 输入还原后缩放至 224×224，并使用官方 ImageNet 均值和标准差归一化。
- [x] Dashboard 新建实验页面增加权重模式选择，并按模型能力禁用无效选项。
- [x] 权重未缓存时在预检和训练日志中明确提示下载需求与缓存路径；只有显式选择 `pretrained` 才会联网下载。
- [x] 验证预训练模式和从头训练检查点都能在不重复下载的情况下安全恢复。
- [x] 四个官方权重（MobileNetV3、EfficientNet-B0、ConvNeXt-Tiny、ViT-B/16）已下载并通过真实 GPU 基准测试。
- [x] MobileNetV3-Small 预训练完整流程已真实跑通（基线、扫描、修复、对比），见下方实测。
- [ ] EfficientNet-B0、ConvNeXt-Tiny、ViT-B/16 尚未跑完整流程；按下表实测耗时，后两者在 6 GB 显卡上不现实。

推荐预训练冒烟命令：

```powershell
python -m failure_atlas.agent run --stage baseline --model mobilenet_v3 --dataset cifar10 --weight-mode pretrained --baseline-epochs 1 --batch-size 64 --run-id e2-mobile-pretrained
python -m failure_atlas.agent run --stage baseline --model efficientnet_b0 --dataset cifar10 --weight-mode pretrained --baseline-epochs 1 --batch-size 64 --run-id e2-efficient-pretrained
python -m failure_atlas.agent run --stage baseline --model convnext_tiny --dataset cifar10 --weight-mode pretrained --baseline-epochs 1 --batch-size 32 --run-id e2-convnext-pretrained
python -m failure_atlas.agent run --stage baseline --model vit_b16 --dataset cifar10 --baseline-epochs 1 --batch-size 16 --run-id e2-vit-b16-pretrained
```

## E2 实测（RTX 3060 Laptop 6 GB，fp32，批量 32/16 等见表）

训练和推理的每步耗时与显存峰值，输入为上采样到 224×224 的 CIFAR 图像：

| 模型 | 训练安全批量 | 每 epoch（4.5 万张） | 训练显存峰值 | 测试集一遍前向 | 完整流程估计 |
|---|---:|---:|---:|---:|---|
| MobileNetV3-Small | 64 | 约 0.5 分钟（含数据加载约 1.1） | 1.06 GB | 1.9 秒（纯前向） | **实测 35.5 分钟** |
| EfficientNet-B0 | 32 | 3.1 分钟 | 2.75 GB | 9.8 秒 | 约 1 小时以上 |
| ConvNeXt-Tiny | 32 | 19.7 分钟 | 3.63 GB | 76 秒 | 约 8 小时 |
| ViT-B/16 | 16 | 27 分钟 | 2.92 GB | 125 到 165 秒 | 12 小时以上 |

- **显存溢出不会报错。** 批量超过显存时驱动悄悄使用共享系统内存，速度下降一个数量级以上：
  EfficientNet-B0 批量 64 为 1.69 秒/步（批量 32 为 0.13 秒），ConvNeXt-Tiny 批量 64 为 22 秒/步
  （批量 32 为 0.84 秒）。因此训练、修复、扫描和对比前都会用实测的一步显存峰值自动缩小批量
  （`fit_batch_size`），不依赖 OOM 报错。
- **学习率。** MobileNetV3 预训练 3 个 epoch 的验证准确率：1e-3 为 88.7%（最后一轮反而下降），
  3e-4 为 92.7%，**1e-4 为 94.1%**，3e-5 为 92.2%（仍在上升）；从头训练 1e-3 只有 50.9%。
  所以预训练微调使用配置学习率的 0.1 倍。只测了一个模型、一个种子。
- **MobileNetV3 完整流程（`mobilenet-pretrained-p0`）**：基线测试准确率 93.84%，修复后 94.03%，ECE 0.0355 到 0.0272；
  基线约 12.5 分钟，扫描约 8 分钟，修复约 3 分钟，对比约 12 分钟，合计约 35.5 分钟。
- **扫描比纯前向慢得多**（8 分钟对 81 遍 × 1.9 秒 ≈ 2.6 分钟），因为每一遍都重新从磁盘解码并转换 1 万张图。
  把测试集预先缓存为张量可明显加速，尚未做。
- 权重缓存在默认的 `~/.cache/torch` 不可写的环境里，可设置环境变量 `TORCH_HOME`（例如项目内的 `.torch-cache/`，已加入 `.gitignore`）。

## E3：统一对比实验

- [ ] 固定 CIFAR-10 数据划分、随机种子和训练预算。
- [ ] 记录 Accuracy、Macro-F1、ECE、参数量和推理速度。
- [ ] 对五类扰动运行相同校准网格。
- [ ] 比较干净性能、轻度/明显/严重失效边界和端点性能。
- [ ] 同时展示修复收益和退化项。
- [ ] 不把不同训练预算或不同权重模式直接混合排名。
- [ ] 生成阶段 E 汇总 Markdown、JSON 和对比图。

## 本轮自动验证

- [x] 新模型定向测试：20 项通过。
- [x] 四个模型选择均通过兼容性验证。
- [x] 四个模型均生成完整五步智能体计划。
- [x] 权重模式定向测试：31 项通过。
- [x] 完整项目测试：103 项通过。
