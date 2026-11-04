"""Build the committed model/dataset compatibility report."""

from failure_atlas.catalog_reporting import write_compatibility_report


if __name__ == "__main__":
    path = write_compatibility_report("docs/compatibility-matrix.md")
    print(f"Compatibility report written to: {path}")
