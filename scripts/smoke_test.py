import argparse

from backend import collections, db, embeddings, retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test retrieval.")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    collection = collections.get_collection_by_id(args.collection)
    if not collection:
        collection = collections.get_collection_by_name(args.collection)
    if not collection:
        raise SystemExit("Collection not found")

    conn = db.get_connection(collections.get_collection_db_path(collection["id"]))
    index = embeddings.load_index(collections.get_collection_faiss_path(collection["id"]))
    results = retrieval.retrieve(conn, args.query, "hybrid", 5, index, True)
    conn.close()
    for item in results:
        print(f"{item['filename']} p.{item['page_num']} | {item['snippet']}")


if __name__ == "__main__":
    main()
