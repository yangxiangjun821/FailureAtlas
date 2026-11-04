from failure_atlas.catalog_reporting import compatibility_markdown


def test_compatibility_report_contains_registered_models_and_datasets() -> None:
    report = compatibility_markdown()
    assert "FashionCNN" in report
    assert "LeNet" in report
    assert "ResNet-18" in report
    assert "MobileNetV3-Small" in report
    assert "EfficientNet-B0" in report
    assert "ConvNeXt-Tiny" in report
    assert "ViT-CIFAR" in report
    assert "Fashion-MNIST" in report
    assert "MNIST" in report
    assert "CIFAR-10" in report
    assert "3×32×32" in report
