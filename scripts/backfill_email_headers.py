import argparse

from backend import collections, db
from backend.indexer import _extract_headers, _parse_names_and_emails


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill email headers from existing pages.")
    parser.add_argument("--collection", required=True, help="Collection name or ID")
    args = parser.parse_args()

    collection = collections.get_collection_by_id(args.collection)
    if not collection:
        collection = collections.get_collection_by_name(args.collection)
    if not collection:
        raise SystemExit("Collection not found")

    conn = db.get_connection(collections.get_collection_db_path(collection["id"]))
    db.init_db(conn)
    docs = conn.execute("SELECT id FROM docs;").fetchall()
    for doc in docs:
        doc_id = doc["id"]
        conn.execute("DELETE FROM email_headers WHERE doc_id = ?;", (doc_id,))
        pages = conn.execute(
            "SELECT page_num, text FROM pages WHERE doc_id = ? ORDER BY page_num;",
            (doc_id,),
        ).fetchall()
        email_rows = []
        for row in pages:
            headers = _extract_headers(row["text"] or "")
            if not headers:
                continue
            from_name, from_email = _parse_names_and_emails(headers.get("from", ""))
            to_name, to_email = _parse_names_and_emails(headers.get("to", ""))
            cc_name, cc_email = _parse_names_and_emails(headers.get("cc", ""))
            snippet = (row["text"] or "")[:400].replace("\n", " ").strip()
            email_rows.append(
                (
                    doc_id,
                    row["page_num"],
                    from_email,
                    to_email,
                    cc_email,
                    from_name,
                    to_name,
                    cc_name,
                    headers.get("subject"),
                    headers.get("date"),
                    snippet,
                )
            )
        db.insert_email_headers(conn, email_rows)
        conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
