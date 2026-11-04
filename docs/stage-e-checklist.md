# 阶段 E 检查清单：现代视觉模型

更新日期：2026-11-04

## E1：模型代码接入

- [x] MobileNetV3-Small 已注册为 `mobilenet_v3`。
- [x] EfficientNet-B0 已注册为 `efficientnet_b0`。
- [x] ConvNeXt-Tiny 已注册为 `convnext_tiny`。
- [x] 轻量 CIFAR ViT 已注册为 `vit`。
- [x] 四个模型均绑定 CIFAR-10、五类图像扰动和 10 类输出。
- [x] 四个模型均通过统一 `ModelAdapter` 接入，不修改智能体主流程。
- [x] 四个模型均能接受 `2×3×32×32` 输入并输出 `2×10` logits。
- [x] Dashboard 模型目录和兼容矩阵可从注册表自动读取新模型。
- [x] `baseline → scan → repair → compare` 五步流水线均能为新模型生成计划。

当前参数量：

| 模型 | 参数量 | 当前模式 |
|---|---:|---|
| MobileNetV3-Small | 1,528,106 | 从头训练 |
| EfficientNet-B0 | 4,020,358 | 从头训练 |
| ConvNeXt-Tiny | 27,827,818 | 从头训练 |
| ViT-CIFAR | 10,695,562 | 从头训练 |

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

- [ ] 为四个模型增加 `scratch` / `pretrained` 权重模式。
- [ ] 使用 TorchVision 官方权重枚举，不使用不明来源权重。
- [ ] 保存权重来源、版本和预处理配置到运行元数据。
- [ ] 适配 ImageNet 预训练输入尺寸与归一化策略。
- [ ] Dashboard 新建实验页面增加权重模式选择。
- [ ] 离线无权重缓存时给出明确提示，不静默联网下载。
- [ ] 验证预训练与从头训练检查点都能安全恢复。

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
- [x] 完整项目测试：76 项通过。
