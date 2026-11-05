# FailureAtlas Model Compatibility Matrix

Generated from the model and dataset registries. A check means the input shape, task, adapter, and preprocessing contract are compatible.

| Model | Fashion-MNIST | MNIST | CIFAR-10 |
|---|---:|---:|---:|
| FashionCNN | ✓ | ✓ | — |
| LeNet | ✓ | ✓ | — |
| ResNet-18 | — | — | ✓ |
| MobileNetV3-Small | — | — | ✓ |
| EfficientNet-B0 | — | — | ✓ |
| ConvNeXt-Tiny | — | — | ✓ |
| ViT-CIFAR | — | — | ✓ |
| ViT-B/16 (ImageNet) | — | — | ✓ |

## Model baselines

| ID | Architecture | Input | Parameters | Compatible datasets |
|---|---|---:|---:|---|
| `fashion_cnn` | compact_cnn | 1×28×28 | 421,834 | fashion_mnist, mnist |
| `lenet` | classic_cnn | 1×28×28 | 44,426 | mnist, fashion_mnist |
| `resnet18` | residual_cnn | 3×32×32 | 11,173,962 | cifar10 |
| `mobilenet_v3` | mobile_cnn | 3×32×32 | 1,528,106 | cifar10 |
| `efficientnet_b0` | compound_scaled_cnn | 3×32×32 | 4,020,358 | cifar10 |
| `convnext_tiny` | modern_convnet | 3×32×32 | 27,827,818 | cifar10 |
| `vit` | vision_transformer | 3×32×32 | 10,695,562 | cifar10 |
| `vit_b16` | vision_transformer_b16 | 3×32×32 | 85,806,346 | cifar10 |

## Dataset preprocessing

| ID | Train / test | Input | Mean | Std | Perturbations |
|---|---:|---:|---|---|---|
| `fashion_mnist` | 60,000 / 10,000 | 1×28×28 | [0.286] | [0.353] | rotation, gaussian_noise, blur, occlusion, brightness |
| `mnist` | 60,000 / 10,000 | 1×28×28 | [0.1307] | [0.3081] | rotation, gaussian_noise, blur, occlusion, brightness |
| `cifar10` | 50,000 / 10,000 | 3×32×32 | [0.4914, 0.4822, 0.4465] | [0.247, 0.2435, 0.2616] | rotation, gaussian_noise, blur, occlusion, brightness |

## Baseline commands

```powershell
python -m scripts.train_baseline --model fashion_cnn --dataset fashion_mnist
python -m scripts.train_baseline --model lenet --dataset mnist
python -m scripts.train_baseline --model resnet18 --dataset cifar10
python -m scripts.train_baseline --model mobilenet_v3 --dataset cifar10
python -m scripts.train_baseline --model efficientnet_b0 --dataset cifar10
python -m scripts.train_baseline --model convnext_tiny --dataset cifar10
python -m scripts.train_baseline --model vit --dataset cifar10
python -m scripts.train_baseline --model vit_b16 --dataset cifar10
```

Weights and generated metrics remain local under `artifacts/`.
