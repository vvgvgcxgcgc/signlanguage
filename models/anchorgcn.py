"""AnchorGCN: CTR-GCN that knows where the hands are relative to the face and torso.

The other graph models only see where a hand is through xyz around the
shoulder midpoint and through message passing along the skeleton, where a
fingertip sits ten hops from the mouth. Here every hand point is described
directly against eight anchors built from real landmarks: both ears, the eye
center, the mouth center, the head center, the neck, and both shoulders.

Everything runs inside the model on the (B, T, J, 4) clip that
preprocess.dataset already emits, so the data pipeline and every other model
stay exactly as they are, and the scale and rotation augmentation is applied
before the relative geometry is measured.

Measured on VSL400, MediaPipe Pose drops all 34 body landmarks together in
0.2% of frames, so the threat to an anchor is not a hole but a landmark that
is present and wrong. The face is therefore repaired as a rigid body: a per
clip template of six face points is fit to every frame with a robust
similarity transform, frames where the fit is poor are treated as missing,
and missing frames take the transform interpolated from their neighbours.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ctrgcn import CTRGCNBlock
from models.embed import MaskedSkeletonEmbed, downsample_mask, init_weights, masked_global_pool
from models.graph import BODY_LANDMARKS, SkeletonGraph
from models.stgcn import BLOCKS

FACE_POINTS = ("leftEar", "rightEar", "leftEye", "rightEye", "mouthLeft", "mouthRight")
ANCHOR_NAMES = (
    "leftEar", "rightEar", "eyeCenter", "mouthCenter", "headCenter",
    "neck", "leftShoulder", "rightShoulder",
)
# Each anchor as a weighted mean of graph joints, used to read anchor features.
ANCHOR_SOURCES = {
    "leftEar": ("leftEar",),
    "rightEar": ("rightEar",),
    "eyeCenter": ("leftEye", "rightEye"),
    "mouthCenter": ("mouthLeft", "mouthRight"),
    "headCenter": ("leftEar", "rightEar"),
    "neck": ("neck",),
    "leftShoulder": ("leftShoulder",),
    "rightShoulder": ("rightShoulder",),
}
FACE_ANCHORS = 5
HAND_TIPS = ("wrist", "thumbTip", "indexTip", "middleTip", "ringTip", "littleTip")
HAND_MCPS = ("indexMCP", "middleMCP", "ringMCP", "littleMCP")
# The MCP centroid has no joint of its own, so its features land on middleMCP.
CENTROID_HOST = "middleMCP"
POINTS_PER_HAND = len(HAND_TIPS) + 1

# BLOCKS without the last temporal stride, so the head sees 32 positions.
BLOCKS_T32 = BLOCKS[:6] + tuple((channels, 1) for channels, _stride in BLOCKS[6:])
HEADS = ("mean", "attentive", "transformer")

RBF_SIGMAS = (0.15, 0.4, 1.0)
# 4 local coordinates, RBF per anchor, the time derivative of the 4
# coordinates and of the 8 distances, and the face and body confidence.
HAND_CHANNELS = 4 + len(ANCHOR_NAMES) * len(RBF_SIGMAS) + 4 + len(ANCHOR_NAMES) + 2
# Offset in the body frame, then RBF.
PAIR_CHANNELS = 2 + len(RBF_SIGMAS)

EPS = 1e-6


def _cmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Complex product of 2D vectors stored as (..., 2)."""
    real = left[..., 0] * right[..., 0] - left[..., 1] * right[..., 1]
    imag = left[..., 0] * right[..., 1] + left[..., 1] * right[..., 0]
    return torch.stack((real, imag), dim=-1)


def _conj(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((value[..., 0], -value[..., 1]), dim=-1)


def _cdiv(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Complex quotient. Maps a point into the frame whose unit vector is `right`."""
    denominator = (right * right).sum(dim=-1, keepdim=True).clamp_min(EPS)
    return _cmul(left, _conj(right)) / denominator


def fill_gaps(values: torch.Tensor, good: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate `values` (B, T, ...) across frames where `good` (B, T) is False.

    Interior gaps mix the nearest good frame on each side, edge gaps hold the
    nearest good frame, and a clip with no good frame comes back as zeros.
    Also returns the distance in frames to the nearest good frame, inf when
    the clip has none.
    """
    batch, frames = good.shape
    steps = torch.arange(frames, device=good.device).expand(batch, frames)
    previous = torch.where(good, steps, torch.full_like(steps, -1)).cummax(dim=1).values
    following = torch.where(good, steps, torch.full_like(steps, frames))
    following = following.flip(1).cummin(dim=1).values.flip(1)
    has_previous = previous >= 0
    has_following = following < frames

    span = (following - previous).clamp_min(1).to(values.dtype)
    weight = torch.where(
        has_previous & has_following,
        (steps - previous).to(values.dtype) / span,
        torch.where(has_previous, torch.zeros_like(span), torch.ones_like(span)),
    )
    shape = (batch, frames) + (1,) * (values.ndim - 2)
    low = values.gather(1, previous.clamp(0, frames - 1).view(shape).expand_as(values))
    high = values.gather(1, following.clamp(0, frames - 1).view(shape).expand_as(values))
    weight = weight.view(shape)
    filled = low * (1.0 - weight) + high * weight
    filled = filled * good.any(dim=1).view((batch,) + (1,) * (values.ndim - 1)).to(values.dtype)

    infinity = torch.full_like(steps, frames * 4, dtype=torch.float32)
    distance = torch.minimum(
        torch.where(has_previous, (steps - previous).float(), infinity),
        torch.where(has_following, (following - steps).float(), infinity),
    )
    distance = torch.where(good.any(dim=1, keepdim=True), distance, torch.full_like(distance, float("inf")))
    return filled, distance


def _confidence(distance: torch.Tensor, decay: float) -> torch.Tensor:
    """1 on a good frame, decaying with the distance to one, 0 when the clip has none."""
    return torch.exp(-distance / decay)


class FaceFit(NamedTuple):
    scale_rotation: torch.Tensor  # (B, T, 2): template unit vector in the image
    translation: torch.Tensor  # (B, T, 2): template origin, the head center, in the image
    template: torch.Tensor  # (B, 1, 6, 2) in template coordinates, ears at (-0.5, 0) and (0.5, 0)
    confidence: torch.Tensor  # (B, T)


def fit_face(
    points: torch.Tensor,
    valid: torch.Tensor,
    repair: bool,
    iterations: int = 3,
    robust_scale: float = 0.1,
    max_residual: float = 0.15,
    max_scale_deviation: float = 0.15,
    decay: float = 4.0,
) -> FaceFit:
    """Fit the six face points of every frame to one template per clip.

    `points` is (B, T, 6, 2) in FACE_POINTS order and `valid` is (B, T, 6).
    Template coordinates put the head center at the origin and the ear-to-ear
    vector at unit length along the first axis. With `repair` off, each frame
    uses its own ears and its own face points as the template, so only the
    gap filling remains.
    """
    ears_ok = valid[..., 0] & valid[..., 1]
    ear_vector = points[:, :, 1] - points[:, :, 0]
    ears_ok = ears_ok & (ear_vector.norm(dim=-1) > EPS)
    head_center = (points[:, :, 0] + points[:, :, 1]) * 0.5
    canonical = _cdiv(points - head_center.unsqueeze(2), ear_vector.unsqueeze(2))

    if not repair:
        good = ears_ok & valid.all(dim=-1)
        frame = torch.cat((ear_vector, head_center), dim=-1)
        filled, distance = fill_gaps(frame, good)
        template, _ = fill_gaps(canonical, good)
        return FaceFit(filled[..., :2], filled[..., 2:], template, _confidence(distance, decay))

    usable = (valid & ears_ok.unsqueeze(-1)).unsqueeze(-1)
    template = torch.nanmedian(
        torch.where(usable, canonical, torch.full_like(canonical, float("nan"))), dim=1, keepdim=True
    ).values
    template_ok = ~torch.isnan(template).any(dim=-1)
    template = torch.nan_to_num(template, nan=0.0)

    weight = (valid & template_ok).to(points.dtype)
    scale_rotation = ear_vector
    translation = head_center
    for _ in range(iterations):
        total = weight.sum(dim=-1, keepdim=True).clamp_min(EPS)
        template_mean = (weight.unsqueeze(-1) * template).sum(dim=2) / total
        point_mean = (weight.unsqueeze(-1) * points).sum(dim=2) / total
        centered_template = template - template_mean.unsqueeze(2)
        centered_points = points - point_mean.unsqueeze(2)
        numerator = (weight.unsqueeze(-1) * _cmul(_conj(centered_template), centered_points)).sum(dim=2)
        denominator = (weight * (centered_template * centered_template).sum(dim=-1)).sum(dim=2)
        scale_rotation = numerator / denominator.clamp_min(EPS).unsqueeze(-1)
        translation = point_mean - _cmul(scale_rotation, template_mean)
        predicted = _cmul(scale_rotation.unsqueeze(2), template) + translation.unsqueeze(2)
        scale = scale_rotation.norm(dim=-1, keepdim=True).clamp_min(EPS)
        residual = (predicted - points).norm(dim=-1) / scale
        weight = (valid & template_ok).to(points.dtype) / (1.0 + (residual / robust_scale) ** 2)

    observed = (valid & template_ok).to(points.dtype)
    count = observed.sum(dim=-1)
    mean_residual = (observed * residual).sum(dim=-1) / count.clamp_min(1.0)
    scale = scale_rotation.norm(dim=-1)
    fitted = count >= 3
    median_scale = torch.nanmedian(
        torch.where(fitted, scale, torch.full_like(scale, float("nan"))), dim=1, keepdim=True
    ).values
    deviation = (scale / median_scale.clamp_min(EPS) - 1.0).abs()
    good = fitted & (mean_residual <= max_residual) & (deviation <= max_scale_deviation)
    good = good & ~torch.isnan(median_scale)

    frame = torch.cat((scale_rotation, translation), dim=-1)
    filled, distance = fill_gaps(frame, good)
    return FaceFit(filled[..., :2], filled[..., 2:], template, _confidence(distance, decay))


class AnchorGeometry(NamedTuple):
    hand: torch.Tensor  # (B, HAND_CHANNELS, T, 14)
    hand_mask: torch.Tensor  # (B, 1, T, 14)
    pair: torch.Tensor  # (B, PAIR_CHANNELS, T, 14, 8)
    pair_weight: torch.Tensor  # (B, 1, T, 14, 8): hand mask times anchor confidence


class AnchorRelativeFeatures(nn.Module):
    """(B, T, J, 4) clip to hand-versus-anchor geometry.

    Every quantity is 2D. Hand z in the clip is wrist-relative while body z is
    hip-relative, so a 3D distance between a fingertip and the nose would mix
    two unrelated depth origins. Computed in float32 without autocast.
    """

    def __init__(self, graph: SkeletonGraph, repair_face: bool = True, decay: float = 4.0) -> None:
        super().__init__()
        index_of = {name: index for index, name in enumerate(graph.names)}
        self.repair_face = bool(repair_face)
        self.decay = float(decay)
        self.register_buffer(
            "face_index", torch.tensor([index_of[name] for name in FACE_POINTS], dtype=torch.long)
        )
        self.register_buffer(
            "shoulder_index",
            torch.tensor([index_of["leftShoulder"], index_of["rightShoulder"]], dtype=torch.long),
        )
        sources: list[list[int]] = []
        hosts: list[int] = []
        for side in ("_0", "_1"):
            for name in HAND_TIPS:
                sources.append([index_of[name + side]] * len(HAND_MCPS))
                hosts.append(index_of[name + side])
            sources.append([index_of[name + side] for name in HAND_MCPS])
            hosts.append(index_of[CENTROID_HOST + side])
        self.register_buffer("hand_sources", torch.tensor(sources, dtype=torch.long))
        self.register_buffer("host_index", torch.tensor(hosts, dtype=torch.long))
        self.register_buffer("sigmas", torch.tensor(RBF_SIGMAS, dtype=torch.float32))

    @property
    def num_points(self) -> int:
        return int(self.host_index.numel())

    def forward(self, clip: torch.Tensor) -> AnchorGeometry:
        xy = clip[..., :2].float()
        valid = clip[..., 3] > 0

        face = fit_face(
            xy[:, :, self.face_index],
            valid[:, :, self.face_index],
            repair=self.repair_face,
            decay=self.decay,
        )
        face_unit = face.scale_rotation
        face_origin = face.translation
        repaired = _cmul(face_unit.unsqueeze(2), face.template) + face_origin.unsqueeze(2)
        eye_center = (repaired[:, :, 2] + repaired[:, :, 3]) * 0.5
        mouth_center = (repaired[:, :, 4] + repaired[:, :, 5]) * 0.5

        shoulders = xy[:, :, self.shoulder_index]
        shoulder_vector = shoulders[:, :, 1] - shoulders[:, :, 0]
        body_good = valid[:, :, self.shoulder_index].all(dim=-1) & (shoulder_vector.norm(dim=-1) > EPS)
        shoulders, body_distance = fill_gaps(shoulders, body_good)
        body_confidence = _confidence(body_distance, self.decay)
        neck = (shoulders[:, :, 0] + shoulders[:, :, 1]) * 0.5
        body_unit = shoulders[:, :, 1] - shoulders[:, :, 0]

        anchors = torch.stack(
            (
                repaired[:, :, 0], repaired[:, :, 1], eye_center, mouth_center, face_origin,
                neck, shoulders[:, :, 0], shoulders[:, :, 1],
            ),
            dim=2,
        )
        anchor_confidence = torch.cat(
            (
                face.confidence.unsqueeze(-1).expand(-1, -1, FACE_ANCHORS),
                body_confidence.unsqueeze(-1).expand(-1, -1, len(ANCHOR_NAMES) - FACE_ANCHORS),
            ),
            dim=-1,
        )

        hand = xy[:, :, self.hand_sources].mean(dim=3)
        hand_mask = valid[:, :, self.hand_sources].all(dim=-1).float()

        face_local = _cdiv(hand - face_origin.unsqueeze(2), face_unit.unsqueeze(2))
        template_eye = (face.template[:, :, 2] + face.template[:, :, 3]) * 0.5
        template_mouth = (face.template[:, :, 4] + face.template[:, :, 5]) * 0.5
        eye_to_mouth = template_mouth[..., 1] - template_eye[..., 1]
        # The sign of eye_to_mouth absorbs whether the camera shows the left ear
        # on the right, so v grows toward the mouth either way.
        sign = torch.where(eye_to_mouth < 0, -torch.ones_like(eye_to_mouth), torch.ones_like(eye_to_mouth))
        eye_to_mouth = sign * eye_to_mouth.abs().clamp_min(0.05)
        face_u = face_local[..., 0]
        face_v = (face_local[..., 1] - template_eye[..., 1].unsqueeze(-1)) / eye_to_mouth.unsqueeze(-1)

        body_local = _cdiv(hand - neck.unsqueeze(2), body_unit.unsqueeze(2))
        head_side = _cdiv(face_origin - neck, body_unit)[..., 1]
        down = torch.where(
            head_side.median(dim=1, keepdim=True).values > 0,
            -torch.ones_like(head_side[:, :1]),
            torch.ones_like(head_side[:, :1]),
        )
        body_u = body_local[..., 0]
        body_v = body_local[..., 1] * down.unsqueeze(-1)

        coordinates = torch.stack((face_u, face_v, body_u, body_v), dim=-1).clamp(-8.0, 8.0)
        face_present = (face.confidence > 0).float().unsqueeze(-1)
        body_present = (body_confidence > 0).float().unsqueeze(-1)
        coordinates = coordinates * torch.cat(
            (face_present, face_present, body_present, body_present), dim=-1
        ).unsqueeze(2)

        body_scale = body_unit.norm(dim=-1).clamp_min(EPS)
        offset = hand.unsqueeze(3) - anchors.unsqueeze(2)
        distance = offset.norm(dim=-1) / body_scale[:, :, None, None]
        anchor_present = (anchor_confidence > 0).float().unsqueeze(2)
        rbf = torch.exp(-(distance.unsqueeze(-1) / self.sigmas) ** 2) * anchor_present.unsqueeze(-1)

        moving = hand_mask[:, 1:] * hand_mask[:, :-1]
        coordinate_motion = torch.zeros_like(coordinates)
        coordinate_motion[:, 1:] = (coordinates[:, 1:] - coordinates[:, :-1]) * moving.unsqueeze(-1)
        distance_motion = torch.zeros_like(distance)
        distance_motion[:, 1:] = (
            (distance[:, 1:] - distance[:, :-1]) * moving.unsqueeze(-1) * anchor_present[:, 1:]
        )

        points = hand.shape[2]
        hand_features = torch.cat(
            (
                coordinates,
                rbf.flatten(3),
                coordinate_motion,
                distance_motion,
                face.confidence.unsqueeze(-1).expand(-1, -1, points).unsqueeze(-1),
                body_confidence.unsqueeze(-1).expand(-1, -1, points).unsqueeze(-1),
            ),
            dim=-1,
        ) * hand_mask.unsqueeze(-1)

        body_offset = _cdiv(offset, body_unit[:, :, None, None])
        pair = torch.cat((body_offset.clamp(-8.0, 8.0), rbf), dim=-1)
        pair_weight = hand_mask.unsqueeze(-1) * anchor_confidence.unsqueeze(2)
        pair = pair * pair_weight.unsqueeze(-1)

        return AnchorGeometry(
            hand=hand_features.permute(0, 3, 1, 2).contiguous(),
            hand_mask=hand_mask.unsqueeze(1),
            pair=pair.permute(0, 4, 1, 2, 3).contiguous(),
            pair_weight=pair_weight.unsqueeze(1),
        )


class AnchorRelativeEmbed(nn.Module):
    """MaskedSkeletonEmbed plus the anchor geometry projected onto the hand joints."""

    def __init__(
        self,
        graph: SkeletonGraph,
        out_channels: int,
        repair_face: bool = True,
    ) -> None:
        super().__init__()
        self.skeleton = MaskedSkeletonEmbed(
            graph.num_joints, out_channels, torch.from_numpy(graph.parents)
        )
        self.geometry = AnchorRelativeFeatures(graph, repair_face=repair_face)
        self.project = nn.Sequential(
            nn.Conv2d(HAND_CHANNELS, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.out_channels = int(out_channels)
        self.num_joints = int(graph.num_joints)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, AnchorGeometry]:
        embedded, mask = self.skeleton(clip)
        geometry = self.geometry(clip)
        relative = self.project(geometry.hand) * geometry.hand_mask
        embedded = embedded.index_add(3, self.geometry.host_index, relative.to(embedded.dtype))
        return embedded, mask, geometry


class AnchorRelationConv(nn.Module):
    """Per-frame messages from the eight anchors to the fourteen hand points.

    Weights are a softmax over anchors computed in every frame, so a touch on
    the chin at one moment is not averaged with the rest of the clip the way
    the clip-pooled topology of CTR-GCN and AAGCN would. The message carries
    the hand-to-anchor offset, not only the anchor features. The output norm
    starts at zero, so at initialization the block is plain CTR-GCN.
    """

    def __init__(self, channels: int, anchor_weights: torch.Tensor, host_index: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("anchor_weights", anchor_weights.clone())
        self.register_buffer("host_index", host_index.clone())
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.geometry = nn.Sequential(
            nn.Conv2d(PAIR_CHANNELS, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.score = nn.Conv2d(channels, 1, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(
        self, features: torch.Tensor, pair: torch.Tensor, pair_weight: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, frames, _ = features.shape
        points = int(self.host_index.numel())
        anchors = int(self.anchor_weights.shape[1])
        anchor_features = torch.einsum(
            "bctj,ja->bcta", features, self.anchor_weights.to(features.dtype)
        )
        hand_features = features[..., self.host_index]
        relative = self.geometry(pair.flatten(3)).view(batch, channels, frames, points, anchors)
        message = self.value(anchor_features).unsqueeze(3) + relative
        query = self.query(hand_features).unsqueeze(-1)
        logits = self.score(F.gelu(message + query).flatten(3)).view(batch, 1, frames, points, anchors)
        logits = logits.float() + torch.log(pair_weight.float().clamp_min(1e-4))
        present = pair_weight > 0
        logits = logits.masked_fill(~present, -1e4)
        weight = torch.softmax(logits, dim=-1) * present
        update = (message.float() * weight).sum(dim=-1)
        update = self.norm(self.out(update.to(features.dtype)))
        return features.index_add(3, self.host_index, update.to(features.dtype))


def _pool_pairs(
    pair: torch.Tensor, pair_weight: torch.Tensor, stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow a strided block along time with a weight-aware average."""
    if stride == 1:
        return pair, pair_weight
    batch, channels, frames, points, anchors = pair.shape
    kernel = (stride, 1)
    weight = F.avg_pool2d(pair_weight.flatten(3), kernel, stride=kernel)
    total = F.avg_pool2d((pair * pair_weight).flatten(3), kernel, stride=kernel)
    pooled = total / weight.clamp_min(EPS)
    new_frames = int(pooled.shape[2])
    return (
        pooled.view(batch, channels, new_frames, points, anchors),
        weight.view(batch, 1, new_frames, points, anchors),
    )


def _part_indices(graph: SkeletonGraph) -> tuple[tuple[int, ...], ...]:
    """Left hand, right hand, face, then the rest of the body."""
    face_names = set(BODY_LANDMARKS[: BODY_LANDMARKS.index("mouthRight") + 1])
    left = tuple(index for index, name in enumerate(graph.names) if name.endswith("_0"))
    right = tuple(index for index, name in enumerate(graph.names) if name.endswith("_1"))
    face = tuple(index for index, name in enumerate(graph.names) if name in face_names)
    taken = set(left) | set(right) | set(face)
    body = tuple(index for index in range(graph.num_joints) if index not in taken)
    return left, right, face, body


ACTIVITY_CHANNELS = 6
# Point 0 of each hand is the wrist; channel 3 is its body-frame height and
# channels 30 and 31 are the time derivative of its body-frame coordinates.
_WRIST_POINTS = (0, POINTS_PER_HAND)
_BODY_V = 3
_BODY_MOTION = (30, 31)


def _activity(geometry: AnchorGeometry, frames: int) -> torch.Tensor:
    """(B, T', 6): wrist height, wrist speed, and hand presence per hand, at the head rate.

    This tells the head whether a frame is rest, raise, or signing without
    cutting any frame out, since some pairs differ inside the raise itself.
    """
    wrists = list(_WRIST_POINTS)
    height = geometry.hand[:, _BODY_V][..., wrists]
    motion = geometry.hand[:, list(_BODY_MOTION)][..., wrists]
    speed = motion.norm(dim=1)
    present = geometry.hand_mask[:, 0][..., wrists]
    activity = torch.cat((height, speed, present), dim=-1).transpose(1, 2)
    return F.adaptive_avg_pool1d(activity, frames).transpose(1, 2)


class FrameTokens(nn.Module):
    """One token per frame that keeps the identity of every hand joint.

    A mean over the 21 joints of a hand mixes an extended index finger with an
    extended little finger, which is exactly the handshape difference between
    the pairs the earlier models confuse. Each hand joint is reduced to a few
    channels and the joints are concatenated in order instead. The face and
    the rest of the body only serve as reference and are averaged.
    """

    def __init__(self, graph: SkeletonGraph, channels: int, dim: int, joint_channels: int = 16) -> None:
        super().__init__()
        left, right, face, body = _part_indices(graph)
        for name, indices in zip(("left", "right", "face", "body"), (left, right, face, body)):
            self.register_buffer(f"part_{name}", torch.tensor(indices, dtype=torch.long))
        self.reduce = nn.Conv2d(channels, joint_channels, kernel_size=1)
        self.left = nn.Linear(joint_channels * len(left), dim)
        self.right = nn.Linear(joint_channels * len(right), dim)
        self.missing = nn.Parameter(torch.zeros(2, dim))
        self.project = nn.Linear(2 * dim + 2 * channels + ACTIVITY_CHANNELS, dim)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.missing, std=0.02)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor, activity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced = self.reduce(features)
        hands = []
        for side, (part, linear) in enumerate(((self.part_left, self.left), (self.part_right, self.right))):
            token = linear(reduced[..., part].permute(0, 2, 3, 1).flatten(2))
            present = mask[..., part].amax(dim=-1).transpose(1, 2).to(token.dtype)
            hands.append(token * present + self.missing[side].to(token.dtype) * (1.0 - present))
        references = []
        for part in (self.part_face, self.part_body):
            weight = mask[..., part]
            total = (features[..., part] * weight).sum(dim=-1)
            count = weight.sum(dim=-1).clamp_min(1.0)
            references.append((total / count).transpose(1, 2))
        tokens = torch.cat((*hands, *references, activity.to(hands[0].dtype)), dim=-1)
        frame_mask = mask.amax(dim=-1)[:, 0] > 0
        return self.norm(self.project(tokens)), frame_mask


class AttentiveStatsPool(nn.Module):
    """Attention-weighted mean and standard deviation over frames.

    The weights of the last forward stay on `last_weights`, (B, T'), so a run
    can check that the model looks at the signing frames and not at the
    raise and lower phases every gloss shares.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.last_weights: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(tokens).squeeze(-1).float().masked_fill(~frame_mask, -1e4)
        weight = torch.softmax(logits, dim=-1) * frame_mask
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(EPS)
        self.last_weights = weight.detach()
        values = tokens.float()
        mean = (weight.unsqueeze(-1) * values).sum(dim=1)
        variance = (weight.unsqueeze(-1) * values * values).sum(dim=1) - mean * mean
        return torch.cat((mean, variance.clamp_min(1e-5).sqrt()), dim=-1).to(tokens.dtype)


def _sinusoid(frames: int, dim: int, device: torch.device) -> torch.Tensor:
    """Fixed positional code, so no parameter depends on T."""
    position = torch.arange(frames, device=device, dtype=torch.float32).unsqueeze(1)
    rate = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-9.2103 / dim))
    table = torch.zeros(frames, dim, device=device)
    table[:, 0::2] = torch.sin(position * rate)
    table[:, 1::2] = torch.cos(position * rate)
    return table


class TemporalHead(nn.Module):
    """Frame tokens, then an optional small encoder that knows frame order, then attentive pooling."""

    def __init__(
        self,
        graph: SkeletonGraph,
        channels: int,
        num_classes: int,
        dim: int,
        layers: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokens = FrameTokens(graph, channels, dim)
        self.encoder = None
        if layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False
            )
        self.pool = AttentiveStatsPool(dim)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, features: torch.Tensor, mask: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        tokens, frame_mask = self.tokens(features, mask, activity)
        # A clip with no observed frame would leave a fully masked row, which
        # makes attention return NaN.
        frame_mask = frame_mask.clone()
        frame_mask[:, 0] |= ~frame_mask.any(dim=1)
        if self.encoder is not None:
            tokens = tokens + _sinusoid(tokens.shape[1], tokens.shape[2], tokens.device).to(tokens.dtype)
            tokens = self.encoder(tokens, src_key_padding_mask=~frame_mask)
        return self.classifier(self.pool(tokens, frame_mask))


class AnchorGCN(nn.Module):
    """Input is (B, T, J, 4), output is class logits.

    `relation_blocks` is how many leading blocks carry an AnchorRelationConv
    between their graph and temporal halves. `head` decides how the last
    block becomes a clip vector:

    - "mean" averages over frames and joints; `part_pool` then pools the two
      hands, the face, and the rest of the body separately. With no relation
      blocks and no part pooling this is CTR-GCN fed the anchor features.
    - "attentive" builds one token per frame that keeps every hand joint, and
      pools frames with learned weights, so a few decisive middle frames are
      not averaged away by the raise and lower phases.
    - "transformer" adds a small encoder over those tokens before pooling,
      so the order of handshapes inside the clip counts.
    """

    def __init__(
        self,
        graph: SkeletonGraph,
        num_classes: int,
        embed_channels: int = 64,
        blocks: Sequence[tuple[int, int]] = BLOCKS,
        reduction: int = 8,
        relation_blocks: int = 6,
        part_pool: bool = True,
        repair_face: bool = True,
        head: str = "mean",
        token_dim: int = 256,
        temporal_layers: int = 2,
    ) -> None:
        super().__init__()
        if not 0 <= relation_blocks <= len(blocks):
            raise ValueError(f"relation_blocks must lie in [0, {len(blocks)}], got {relation_blocks}")
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        adjacency = torch.from_numpy(graph.adjacency.copy())
        self.embed = AnchorRelativeEmbed(graph, embed_channels, repair_face=repair_face)

        index_of = {name: index for index, name in enumerate(graph.names)}
        anchor_weights = torch.zeros(graph.num_joints, len(ANCHOR_NAMES))
        for column, name in enumerate(ANCHOR_NAMES):
            sources = ANCHOR_SOURCES[name]
            for source in sources:
                anchor_weights[index_of[source], column] = 1.0 / len(sources)

        channels = embed_channels
        layers = []
        relations = []
        strides = []
        for position, (out_channels, stride) in enumerate(blocks):
            layers.append(CTRGCNBlock(channels, out_channels, adjacency, stride, reduction))
            if position < relation_blocks:
                relations.append(
                    AnchorRelationConv(out_channels, anchor_weights, self.embed.geometry.host_index)
                )
            strides.append(int(stride))
            channels = out_channels
        self.blocks = nn.ModuleList(layers)
        self.relations = nn.ModuleList(relations)
        self.strides = tuple(strides)

        self.head_kind = head
        self.part_pool = bool(part_pool) and head == "mean"
        if head != "mean":
            layers_count = temporal_layers if head == "transformer" else 0
            self.head = TemporalHead(graph, channels, num_classes, token_dim, layers_count)
        elif self.part_pool:
            for name, indices in zip(("left", "right", "face", "body"), _part_indices(graph)):
                self.register_buffer(f"part_{name}", torch.tensor(indices, dtype=torch.long))
            self.head = nn.Linear(channels * 4, num_classes)
        else:
            self.head = nn.Linear(channels, num_classes)
        self.apply(init_weights)
        for relation in self.relations:
            nn.init.zeros_(relation.norm.weight)

    @property
    def frame_weights(self) -> torch.Tensor | None:
        """Attention over the head positions from the last forward, or None for the mean head."""
        if self.head_kind == "mean":
            return None
        return self.head.pool.last_weights

    def _pool(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.part_pool:
            return masked_global_pool(features, mask)
        parts = (self.part_left, self.part_right, self.part_face, self.part_body)
        return torch.cat(
            [masked_global_pool(features[..., part], mask[..., part]) for part in parts], dim=1
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        features, mask, geometry = self.embed(clip)
        pair, pair_weight = geometry.pair, geometry.pair_weight
        for position, (block, stride) in enumerate(zip(self.blocks, self.strides)):
            if position < len(self.relations):
                hidden = self.relations[position](block.gcn(features), pair, pair_weight)
                features = F.relu(block.tcn(hidden) + block.residual(features))
                pair, pair_weight = _pool_pairs(pair, pair_weight, stride)
            else:
                features = block(features)
            mask = downsample_mask(mask, stride)
        if self.head_kind != "mean":
            return self.head(features, mask, _activity(geometry, int(features.shape[2])))
        return self.head(self._pool(features, mask))
