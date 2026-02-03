import argparse
import os

from backend import collections


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a collection.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    item = collections.create_collection(args.name, root)
    print(item["id"])


if __name__ == "__main__":
    main()
