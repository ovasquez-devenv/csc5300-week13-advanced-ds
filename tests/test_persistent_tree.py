"""Path copying keeps old versions and shares untouched nodes."""

import random

from advds.trees import PersistentArray, PersistentSegmentTree, VersionStore, count_unique_nodes


def test_persistent_array_book_trace():
    v0 = PersistentArray([1, 2, 3, 4, 5])
    v1 = v0.set(2, 10)
    assert v0.to_list() == [1, 2, 3, 4, 5]
    assert v1.to_list() == [1, 2, 10, 4, 5]
    v2 = v1.set(4, 20)
    assert v2.to_list() == [1, 2, 10, 4, 20]
    assert v0.get(2) == 3
    # Index 2 sits in the left half, so the right subtree object is shared.
    assert v1.root.right is v0.root.right


def test_persistent_segment_versions_and_branch():
    code = [5, 3, 7, 2, 9, 1, 4, 8]
    store = VersionStore(code)
    v1 = store.update(2, 15)
    v2 = store.update(5, 10)
    v3 = store.update(4, 20, version=v1)
    assert store.query(1, 5, version=0) == sum(code[1:6])
    edited = code.copy()
    edited[2] = 15
    assert store.get_list(v1) == edited
    edited2 = edited.copy()
    edited2[5] = 10
    assert store.get_list(v2) == edited2
    branch = edited.copy()
    branch[4] = 20
    assert store.get_list(v3) == branch
    # v3 was branched from v1, so it does not contain v2's edit.
    assert store.get_list(v3)[5] == 1
    changes = store.diff(v1, v3)
    assert changes == [(4, 9, 20)]


def test_node_sharing_is_cheaper_than_copying():
    store = VersionStore(list(range(32)))
    for index in range(10):
        store.update(index, 100 + index)
    unique = store.unique_nodes()
    copied = store.nodes_if_copied()
    assert unique < copied
    assert count_unique_nodes(store.versions) == unique
    # One edit must not clone the opposite child of the root.
    left_edit = store.versions[0].update(0, 7)
    assert left_edit.root.right is store.versions[0].root.right


def test_random_history_matches_lists():
    rng = random.Random(9)
    n = 40
    current = [rng.randint(0, 20) for _ in range(n)]
    store = VersionStore(current)
    snapshots = [current.copy()]
    for _ in range(25):
        base = rng.randrange(len(snapshots))
        current = snapshots[base].copy()
        index = rng.randrange(n)
        value = rng.randint(0, 30)
        current[index] = value
        new_id = store.update(index, value, version=base)
        snapshots.append(current)
        assert new_id == len(snapshots) - 1
        assert store.get_list(new_id) == current
        left = rng.randrange(n)
        right = rng.randint(left, n - 1)
        assert store.query(left, right, new_id) == sum(current[left : right + 1])


def test_persistent_min():
    tree = PersistentSegmentTree([4, 1, 7, 3], operation="min")
    assert tree.query(0, 3) == 1
    nxt = tree.update(1, 9)
    assert tree.query(0, 3) == 1
    assert nxt.query(0, 3) == 3
    assert nxt.query(1, 1) == 9
