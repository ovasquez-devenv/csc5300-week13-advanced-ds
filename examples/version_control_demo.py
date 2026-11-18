"""A few commits on the persistent file store.

Run from the repository root:

    python examples/version_control_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advds import VersionControl


def main() -> None:
    repo = VersionControl()
    first = repo.commit("initial", {"notes.txt": "alpha beta", "main.py": "print(1)\n"})
    second = repo.commit("change the print", {"notes.txt": "alpha beta", "main.py": "print(2)\n"})
    side = repo.commit(
        "branch off the first commit",
        {"notes.txt": "alpha beta!", "main.py": "print(1)\n"},
        parent=first,
    )
    print("Log:")
    for line in repo.log():
        print(" ", line)
    print("\nHistory:")
    print(repo.history_graph())
    print("\nCheckout of the side branch:", repo.checkout(side))
    print("\nDiff from the first commit to the second:")
    for item in repo.diff(first, second):
        print(f"  {item['file']}: {item['status']} {item['changes']}")
    print("\nDiff from the first commit to the side branch:")
    for item in repo.diff(first, side):
        status = item["status"]
        extra = item["changes"] if item["changes"] else f"len {item['old_len']} -> {item['new_len']}"
        print(f"  {item['file']}: {status} {extra}")


if __name__ == "__main__":
    main()
