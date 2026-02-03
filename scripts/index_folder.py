import argparse
import os

from backend.indexer import index_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDFs in a folder.")
    parser.add_argument("--root", required=True, help="Root folder of PDFs")
    parser.add_argument(
        "--reindex", action="store_true", help="Rebuild the index from scratch"
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    stats = index_folder(root, reindex=args.reindex)
    print(
        f"Indexed: {stats['indexed']} | Skipped: {stats['skipped']} | Errors: {stats['errors']}"
    )


if __name__ == "__main__":
    main()
