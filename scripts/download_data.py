"""Download Fashion-MNIST so later experiments can run offline."""

from failure_atlas.data import build_loaders


if __name__ == "__main__":
    train_loader, test_loader = build_loaders("data", batch_size=128, download=True)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("Fashion-MNIST is cached and ready for offline use.")
