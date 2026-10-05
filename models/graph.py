"""Skeleton graph for the (T, J, 4) clips that preprocess.dataset emits.

Joint order is 34 body landmarks followed by 42 hand landmarks interleaved
left and right, which is the order preprocess.keypoints.LANDMARKS builds.
Edges are declared by landmark name and only then resolved to indices, so one
declaration covers both the full 76-joint skeleton and the 68-joint skeleton
that preprocess.dataset.drop_legs leaves behind.

The landmark names are repeated here instead of imported because
preprocess.keypoints pulls in OpenCV, which training does not need. Call
verify_joint_names once in an environment that has OpenCV to confirm the two
declarations still agree.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

BODY_LANDMARKS = (
    "nose", "leftEyeInner", "leftEye", "leftEyeOuter", "rightEyeInner", "rightEye", "rightEyeOuter",
    "leftEar", "rightEar", "mouthLeft", "mouthRight", "leftShoulder", "rightShoulder",
    "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftPinky", "rightPinky",
    "leftIndex", "rightIndex", "leftThumb", "rightThumb",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle",
    "leftHeel", "rightHeel", "leftFootIndex", "rightFootIndex", "neck",
)
HAND_LANDMARKS = (
    "wrist", "indexTip", "indexDIP", "indexPIP", "indexMCP",
    "middleTip", "middleDIP", "middlePIP", "middleMCP",
    "ringTip", "ringDIP", "ringPIP", "ringMCP",
    "littleTip", "littleDIP", "littlePIP", "littleMCP",
    "thumbTip", "thumbIP", "thumbMP", "thumbCMC",
)
LANDMARKS = BODY_LANDMARKS + tuple(
    name + suffix for name in HAND_LANDMARKS for suffix in ("_0", "_1")
)

# Dropped by preprocess.dataset.drop_legs, which keeps the hips.
LEG_LANDMARKS = (
    "leftKnee", "rightKnee", "leftAnkle", "rightAnkle",
    "leftHeel", "rightHeel", "leftFootIndex", "rightFootIndex",
)

ROOT_NAME = "neck"
N_PARTITIONS = 3

BODY_BONES = (
    ("nose", "leftEyeInner"), ("leftEyeInner", "leftEye"), ("leftEye", "leftEyeOuter"),
    ("leftEyeOuter", "leftEar"),
    ("nose", "rightEyeInner"), ("rightEyeInner", "rightEye"), ("rightEye", "rightEyeOuter"),
    ("rightEyeOuter", "rightEar"),
    ("nose", "mouthLeft"), ("nose", "mouthRight"), ("mouthLeft", "mouthRight"),
    ("nose", "neck"),
    ("neck", "leftShoulder"), ("neck", "rightShoulder"), ("leftShoulder", "rightShoulder"),
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("leftWrist", "leftThumb"), ("leftWrist", "leftIndex"), ("leftWrist", "leftPinky"),
    ("rightWrist", "rightThumb"), ("rightWrist", "rightIndex"), ("rightWrist", "rightPinky"),
    ("leftShoulder", "leftHip"), ("rightShoulder", "rightHip"), ("leftHip", "rightHip"),
)
LEG_BONES = (
    ("leftHip", "leftKnee"), ("leftKnee", "leftAnkle"),
    ("leftAnkle", "leftHeel"), ("leftHeel", "leftFootIndex"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("rightAnkle", "rightHeel"), ("rightHeel", "rightFootIndex"),
)
HAND_BONES = (
    ("wrist", "thumbCMC"), ("thumbCMC", "thumbMP"), ("thumbMP", "thumbIP"), ("thumbIP", "thumbTip"),
    ("wrist", "indexMCP"), ("indexMCP", "indexPIP"), ("indexPIP", "indexDIP"), ("indexDIP", "indexTip"),
    ("wrist", "middleMCP"), ("middleMCP", "middlePIP"), ("middlePIP", "middleDIP"),
    ("middleDIP", "middleTip"),
    ("wrist", "ringMCP"), ("ringMCP", "ringPIP"), ("ringPIP", "ringDIP"), ("ringDIP", "ringTip"),
    ("wrist", "littleMCP"), ("littleMCP", "littlePIP"), ("littlePIP", "littleDIP"),
    ("littleDIP", "littleTip"),
    ("indexMCP", "middleMCP"), ("middleMCP", "ringMCP"), ("ringMCP", "littleMCP"),
)
# Suffix _0 is the left hand and _1 is the right hand, so each hand root is
# wired to the pose wrist on its own side.
CROSS_BONES = (("leftWrist", "wrist_0"), ("rightWrist", "wrist_1"))


@dataclass(frozen=True)
class SkeletonGraph:
    """Everything the models need to know about the joint layout.

    `adjacency` is the ST-GCN spatial configuration: partition 0 links joints
    at the same hop distance from the root, partition 1 points inward, and
    partition 2 points outward. `parents` gives one parent per joint from a
    breadth-first tree rooted at the neck, which is what the bone channel
    subtracts; the root is its own parent so its bone is zero.
    """

    names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    adjacency: np.ndarray
    parents: np.ndarray
    root: int
    is_leg: bool

    @property
    def num_joints(self) -> int:
        return len(self.names)

    @property
    def num_partitions(self) -> int:
        return int(self.adjacency.shape[0])


def joint_names(is_leg: bool) -> tuple[str, ...]:
    """Landmark names in the order the dataset emits them."""
    if is_leg:
        return LANDMARKS
    dropped = set(LEG_LANDMARKS)
    return tuple(name for name in LANDMARKS if name not in dropped)


def _bone_names(is_leg: bool) -> tuple[tuple[str, str], ...]:
    bones = list(BODY_BONES)
    if is_leg:
        bones.extend(LEG_BONES)
    for left, right in HAND_BONES:
        bones.append((f"{left}_0", f"{right}_0"))
        bones.append((f"{left}_1", f"{right}_1"))
    bones.extend(CROSS_BONES)
    return tuple(bones)


def build_edges(is_leg: bool) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Resolve the declared bones to index pairs over the emitted joints."""
    names = joint_names(is_leg)
    index_of = {name: index for index, name in enumerate(names)}
    edges: list[tuple[int, int]] = []
    for left, right in _bone_names(is_leg):
        if left not in index_of:
            raise KeyError(f"bone endpoint is not a known landmark: {left}")
        if right not in index_of:
            raise KeyError(f"bone endpoint is not a known landmark: {right}")
        edges.append((index_of[left], index_of[right]))
    return names, tuple(edges)


def hop_distance(num_joints: int, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Shortest hop count between every pair, inf where no path exists."""
    hops = np.full((num_joints, num_joints), np.inf, dtype=np.float64)
    np.fill_diagonal(hops, 0.0)
    for left, right in edges:
        hops[left, right] = hops[right, left] = 1.0
    for middle in range(num_joints):
        hops = np.minimum(hops, hops[:, middle, None] + hops[None, middle, :])
    return hops


def _bfs_parents(num_joints: int, edges: tuple[tuple[int, int], ...], root: int) -> np.ndarray:
    neighbours: list[list[int]] = [[] for _ in range(num_joints)]
    for left, right in edges:
        neighbours[left].append(right)
        neighbours[right].append(left)
    parents = np.full(num_joints, -1, dtype=np.int64)
    parents[root] = root
    queue = deque([root])
    while queue:
        joint = queue.popleft()
        for neighbour in neighbours[joint]:
            if parents[neighbour] == -1:
                parents[neighbour] = joint
                queue.append(neighbour)
    if int((parents < 0).sum()) > 0:
        unreached = [int(index) for index in np.flatnonzero(parents < 0)]
        raise ValueError(f"joints unreachable from the root: {unreached}")
    return parents


def spatial_adjacency(
    num_joints: int,
    edges: tuple[tuple[int, int], ...],
    root: int,
) -> np.ndarray:
    """Split the one-hop neighbourhood into the three ST-GCN partitions.

    Index the result as `A[k, source, target]` and aggregate as
    `out[target] = sum over source of x[source] * A[k, source, target]`. The
    one-hop neighbourhood, self loops included, is normalized along the source
    axis before the split, so summing the three partitions gives each target a
    set of weights adding to 1. Every model must sum over the source index;
    transposing it silently turns the average into an unnormalized sum and
    swaps the centripetal and centrifugal partitions.
    """
    hops = hop_distance(num_joints, edges)
    if not np.isfinite(hops).all():
        raise ValueError("skeleton graph is disconnected")
    to_root = hops[root]

    neighbour = (hops <= 1).astype(np.float64)
    degree = neighbour.sum(axis=0)
    normalized = neighbour / np.maximum(degree, 1e-12)

    partitions = np.zeros((N_PARTITIONS, num_joints, num_joints), dtype=np.float32)
    for source in range(num_joints):
        for target in range(num_joints):
            if neighbour[source, target] == 0:
                continue
            if to_root[target] == to_root[source]:
                partition = 0
            elif to_root[target] < to_root[source]:
                partition = 1
            else:
                partition = 2
            partitions[partition, source, target] = normalized[source, target]
    return partitions


def build_graph(is_leg: bool = False) -> SkeletonGraph:
    """Graph for the 68-joint set by default, or the full 76 when is_leg."""
    names, edges = build_edges(is_leg)
    num_joints = len(names)
    root = names.index(ROOT_NAME)
    adjacency = spatial_adjacency(num_joints, edges, root)
    parents = _bfs_parents(num_joints, edges, root)
    return SkeletonGraph(
        names=names,
        edges=edges,
        adjacency=adjacency,
        parents=parents,
        root=root,
        is_leg=is_leg,
    )


def verify_joint_names() -> None:
    """Fail if the local landmark order drifted from preprocess.

    Importing preprocess pulls in OpenCV, so this stays out of build_graph and
    is meant for a test or a one-off check.
    """
    from preprocess.dataset import LEG_INDICES, N_JOINTS
    from preprocess.keypoints import LANDMARKS as SOURCE_LANDMARKS

    if tuple(SOURCE_LANDMARKS) != LANDMARKS:
        raise AssertionError("models.graph.LANDMARKS drifted from preprocess.keypoints.LANDMARKS")
    if len(LANDMARKS) != N_JOINTS:
        raise AssertionError(f"expected {N_JOINTS} landmarks, got {len(LANDMARKS)}")
    dropped = tuple(LANDMARKS[index] for index in LEG_INDICES)
    if dropped != LEG_LANDMARKS:
        raise AssertionError(f"LEG_LANDMARKS does not match LEG_INDICES, which selects {dropped}")
