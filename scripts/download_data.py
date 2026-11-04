"""Download the selected dataset so later experiments can run offline."""

import argparse

from failure_atlas.adapters import get_dataset_adapter
from failure_atlas.catalog import validate_selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fashion_cnn")
    parser.add_argument("--dataset", default="fashion_mnist")
    args = parser.parse_args()
    _, dataset_spec, _ = validate_selection(args.model, args.dataset)
    adapter = get_dataset_adapter(args.dataset)
    train_loader, test_loader = adapter.prepare("data", batch_size=128, download=True)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"{dataset_spec.display_name} is cached and ready for offline use.")
