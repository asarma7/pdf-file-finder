import argparse

from backend import collections, indexer


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a collection.")
    parser.add_argument("--collection", required=True, help="Collection name or id")
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    collection = collections.get_collection_by_id(args.collection)
    if not collection:
        collection = collections.get_collection_by_name(args.collection)
    if not collection:
        raise SystemExit("Collection not found")

    stats = indexer.index_collection(collection["id"], reindex=args.reindex)
    print(
        f"Indexed: {stats['indexed']} | Skipped: {stats['skipped']} | Errors: {stats['errors']}"
    )


if __name__ == "__main__":
    main()
