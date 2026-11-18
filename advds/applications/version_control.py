"""A tiny version control store backed by persistent segment trees.

Section 12.4.3 uses a persistent segment tree as the history of a sequence.
Each commit in this module is one snapshot of the working files. A file is a
persistent segment tree of character codes. If the new text has the same
length as the parent file, the commit is a chain of path copying updates, so
unchanged regions stay shared with the parent. ``diff`` walks the two roots
and skips any subtree whose node object is shared, which is the fast path
Amakobe describes for versions that share structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..trees.persistent_tree import PersistentSegmentTree, _SegNode


@dataclass
class Commit:
    commit_id: int
    parent: Optional[int]
    message: str
    files: Dict[str, PersistentSegmentTree] = field(default_factory=dict)


def _tree_from_text(text: str, parent: Optional[PersistentSegmentTree]) -> PersistentSegmentTree:
    values = [ord(char) for char in text]
    if parent is None or parent.n != len(values):
        return PersistentSegmentTree(values)
    current = parent
    previous = parent.to_list()
    for index, value in enumerate(values):
        if previous[index] != value:
            current = current.update(index, value)
    return current


def _text_from_tree(tree: PersistentSegmentTree) -> str:
    return "".join(chr(int(value)) for value in tree.to_list())


def _diff_trees(left: PersistentSegmentTree, right: PersistentSegmentTree) -> List[Tuple[int, str, str]]:
    """Pairs ``(index, old_char, new_char)`` for a shared length file."""
    if left.n != right.n:
        raise ValueError("Structural diff expects equal lengths.")
    if left.n == 0 or left.root is None or right.root is None:
        return []
    changes: List[Tuple[int, str, str]] = []

    def walk(a: _SegNode, b: _SegNode, lo: int, hi: int) -> None:
        if a is b:
            return
        if lo == hi:
            if a.value != b.value:
                changes.append((lo, chr(int(a.value)), chr(int(b.value))))
            return
        mid = (lo + hi) // 2
        assert a.left is not None and a.right is not None
        assert b.left is not None and b.right is not None
        walk(a.left, b.left, lo, mid)
        walk(a.right, b.right, mid + 1, hi)

    walk(left.root, right.root, 0, left.n - 1)
    return changes


class VersionControl:
    """Commit, checkout, and diff a handful of text files.

    Commit ids are integers starting at 0. ``parent`` defaults to the current
    head, and passing an older id branches the way the chapter's version
    control sketch branches off version 1. There is no staging area and no
    merge. One commit stores one version of each file you pass in. Files
    omitted from a commit are deletions relative to the parent.
    """

    def __init__(self) -> None:
        self._commits: List[Commit] = []
        self.head: Optional[int] = None

    def commit(self, message: str, files: Dict[str, str], parent: Optional[int] = None) -> int:
        """Store ``files`` as a new commit and move ``head``. Returns the id."""
        if parent is None:
            parent = self.head
        if parent is not None and not 0 <= parent < len(self._commits):
            raise IndexError(parent)
        parent_commit = None if parent is None else self._commits[parent]
        snapshot: Dict[str, PersistentSegmentTree] = {}
        for name, text in files.items():
            if not isinstance(text, str):
                raise TypeError(f"{name} must be a str, not {type(text).__name__}.")
            previous = None
            if parent_commit is not None and name in parent_commit.files:
                previous = parent_commit.files[name]
            snapshot[name] = _tree_from_text(text, previous)
        commit = Commit(
            commit_id=len(self._commits),
            parent=parent,
            message=message,
            files=snapshot,
        )
        self._commits.append(commit)
        self.head = commit.commit_id
        return commit.commit_id

    def checkout(self, commit_id: int) -> Dict[str, str]:
        """Rebuild file texts from the persistent trees at ``commit_id``."""
        commit = self._get(commit_id)
        return {name: _text_from_tree(tree) for name, tree in sorted(commit.files.items())}

    def diff(self, left_id: int, right_id: int) -> List[dict]:
        """Per file changes from ``left_id`` to ``right_id``.

        Equal length files use the shared node walk. A length change is
        reported as a full replacement, because path copying in this tree
        does not resize a version.
        """
        left = self._get(left_id)
        right = self._get(right_id)
        names = sorted(set(left.files) | set(right.files))
        report: List[dict] = []
        for name in names:
            old_tree = left.files.get(name)
            new_tree = right.files.get(name)
            if old_tree is None and new_tree is not None:
                report.append(
                    {
                        "file": name,
                        "status": "added",
                        "old_len": 0,
                        "new_len": new_tree.n,
                        "changes": [],
                    }
                )
                continue
            if old_tree is not None and new_tree is None:
                report.append(
                    {
                        "file": name,
                        "status": "deleted",
                        "old_len": old_tree.n,
                        "new_len": 0,
                        "changes": [],
                    }
                )
                continue
            assert old_tree is not None and new_tree is not None
            if old_tree.root is new_tree.root:
                report.append(
                    {
                        "file": name,
                        "status": "unchanged",
                        "old_len": old_tree.n,
                        "new_len": new_tree.n,
                        "changes": [],
                    }
                )
                continue
            if old_tree.n != new_tree.n:
                report.append(
                    {
                        "file": name,
                        "status": "replaced",
                        "old_len": old_tree.n,
                        "new_len": new_tree.n,
                        "old_text": _text_from_tree(old_tree),
                        "new_text": _text_from_tree(new_tree),
                        "changes": [],
                    }
                )
                continue
            changes = _diff_trees(old_tree, new_tree)
            report.append(
                {
                    "file": name,
                    "status": "modified" if changes else "unchanged",
                    "old_len": old_tree.n,
                    "new_len": new_tree.n,
                    "changes": changes,
                }
            )
        return report

    def log(self) -> List[str]:
        """One line per commit, oldest first."""
        lines = []
        for commit in self._commits:
            parent = "none" if commit.parent is None else str(commit.parent)
            lines.append(f"c{commit.commit_id} parent {parent}: {commit.message}")
        return lines

    def history_graph(self) -> str:
        """Indent commits under the parent they were branched from."""
        children: Dict[Optional[int], List[int]] = {}
        for commit in self._commits:
            children.setdefault(commit.parent, []).append(commit.commit_id)
        lines: List[str] = []

        def walk(commit_id: int, depth: int) -> None:
            commit = self._commits[commit_id]
            lines.append(f"{'  ' * depth}* c{commit.commit_id}: {commit.message}")
            for child in children.get(commit_id, []):
                walk(child, depth + 1)

        for root in children.get(None, []):
            walk(root, 0)
        return "\n".join(lines)

    def _get(self, commit_id: int) -> Commit:
        if not 0 <= commit_id < len(self._commits):
            raise IndexError(commit_id)
        return self._commits[commit_id]

    def __len__(self) -> int:
        return len(self._commits)

    def __repr__(self) -> str:
        return f"VersionControl(commits={len(self._commits)}, head={self.head})"
