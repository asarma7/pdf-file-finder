#!/usr/bin/env python3
"""Clear email_headers and email_aliases for one or all collections. Other DB data is left intact."""
import argparse
import sys

from backend import collections, db


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear email headers and aliases only (per collection).")
    parser.add_argument(
        "--collection",
        help="Collection ID or name; if omitted, clear all collections.",
    )
    args = parser.parse_args()

    if args.collection:
        coll = collections.get_collection_by_id(args.collection) or collections.get_collection_by_name(
            args.collection
        )
        if not coll:
            print(f"Collection not found: {args.collection}", file=sys.stderr)
            sys.exit(1)
        collections_list = [coll]
    else:
        collections_list = list(collections.list_collections())
        if not collections_list:
            print("No collections found.", file=sys.stderr)
            sys.exit(0)

    for coll in collections_list:
        cid = coll["id"]
        name = coll.get("name", cid)
        path = collections.get_collection_db_path(cid)
        if not path.exists():
            print(f"Skip {name}: no DB at {path}")
            continue
        conn = db.get_connection(path)
        db.init_db(conn)
        db.clear_email_data(conn)
        conn.close()
        print(f"Cleared email_headers and email_aliases for: {name} ({cid})")

    print("Done.")


if __name__ == "__main__":
    main()
