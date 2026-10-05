# Skeleton classifiers

Five gloss classifiers over MediaPipe keypoint clips, plus the skeleton graph
and the input adaptor they share. Everything here is built for the tensors
`preprocess.dataset.KeypointDataset` emits, and nothing here imports OpenCV or
the extraction pipeline, so training only needs `torch`.

```python
from models import build_model

model = build_model("CTR-GCN", num_classes=422, is_leg=False, max_frames=64)
logits = model(clip)          # clip: (B, 64, 68, 4) -> logits: (B, 422)
```

## Input contract

Every model takes one tensor and returns logits.

| | shape | meaning |
|---|---|---|
| input | `(B, T, J, 4)` | channels 0-2 are xyz, channel 3 is validity |
| output | `(B, num_classes)` | unnormalized logits, feed to `CrossEntropyLoss` |

- **`J` is a hard structural choice.** `is_leg=False` gives 68 joints, which is
  what `preprocess.dataset.drop_legs` leaves; `is_leg=True` gives all 76. The
  adjacency buffers, the per-joint missing token, and the flattening
  projections all scale with `J`, so a checkpoint cannot move between the two.
  Prefer 68: the eight extra joints are knee, ankle, heel, and foot, which are
  mostly holes in signing video and would take 10.5% of the global pooling
  mass.
- **`T` is nearly free.** No parameter in `MS-TCN`, `ST-GCN`, `CTR-GCN`, or
  `AAGCN` depends on it, so one checkpoint runs at any length. Only
  `Transformer` is bounded, by the learned positional table sized to
  `max_frames`.
- `is_leg` passed to `build_model` **must match** the flag the dataset was
  built with. There is no way to detect a mismatch from the tensor alone when
  both sides happen to agree on `J`.

## The five models

| name | params (J=68) | params (J=76) | role |
|---|---|---|---|
| `MS-TCN` | 690,742 | 707,254 | cheap temporal-only floor every graph model must beat |
| `ST-GCN` | 3,265,046 | 3,296,662 | honest graph baseline, the number to compare against |
| `CTR-GCN` | 1,640,529 | 1,672,145 | best accuracy per parameter in this family; start here |
| `AAGCN` | 3,919,089 | 3,950,705 | learns its own topology, survives a wrong declared graph |
| `Transformer` | 5,144,502 | 5,177,398 | different failure modes, worth it only in an ensemble |

Counts are measured at `num_classes=422`. The classifier head alone is 108,454
of them, so a different gloss count shifts every row by roughly the same
amount.

### MS-TCN — `mstcn.py`

Joints are embedded per frame, flattened into one channel axis, and pushed
through five residual blocks of two dilated kernel-3 convolutions at rates
1, 2, 4, 8, 16. Receptive field is **125 frames**, so a 64-frame clip is
covered end to end. No graph is involved, which is the point. If a graph model
does not beat this clearly, the graph model has a bug.

### ST-GCN — `stgcn.py`

Nine blocks of graph convolution over the three spatial partitions followed by
a 9x1 temporal convolution. Channels go 64, 64, 64, 128, 128, 128, 256, 256,
256 with stride 2 at blocks 4 and 7, so time runs **64 → 32 → 16**. Each block
scales the shared adjacency by its own learnable `edge_importance`.

### CTR-GCN — `ctrgcn.py`

Same block layout, two parts replaced. Spatially, each partition refines the
shared topology **per output channel** from the pairwise difference between
joint embeddings, so the link between index and middle finger can matter to
the channels describing handshape and mean nothing to the channels describing
where the arm is. Temporally, one 9x1 convolution becomes four parallel views
(two dilated 5x1 convolutions, a max pool, a 1x1), each `out_channels // 4`
wide. That temporal swap is where most of the saving against ST-GCN comes
from. `alpha` starts at zero, so at initialization this is a plain graph
convolution on the declared skeleton.

### AAGCN — `aagcn.py`

ST-GCN backbone where every partition sums three matrices: the declared
graph, a freely learned offset, and one inferred from the clip. A wrong edge
can be unlearned, which is the reason to keep this around at all. Each block
also carries three residual sigmoid gates over joints, frames, and channels.
About 2.4 times CTR-GCN's parameters for the same block layout, so it needs
more data to pay off.

### Transformer — `transformer.py`

Six pre-norm encoder layers, `d_model=256`, eight heads, feedforward 1024,
over flattened per-frame joint embeddings. A learned positional table sized to
`max_frames + 1` replaces a 5000-row sinusoidal buffer. Classification reads a
prepended class token rather than a mean over frames, so a long still pose
cannot wash out the few frames carrying the sign. Frames where nothing was
observed are dropped from attention through the key padding mask; the class
token is always kept so no row can be fully masked.

## The skeleton graph — `graph.py`

Joint order is 34 body landmarks, then 42 hand landmarks **interleaved** left
and right: index 34 is `wrist_0` (left), 35 is `wrist_1` (right), and so on.
This is the one detail that makes hand edges easy to get wrong, which is why
every edge here is declared by landmark name and only then resolved to an
index.

```python
from models import build_graph

graph = build_graph(is_leg=False)   # 68 joints, 76 edges, root is "neck"
graph.adjacency                     # (3, J, J) float32
graph.parents                       # (J,) int64, one BFS parent per joint
```

The landmark names are repeated in this module instead of imported, because
`preprocess.keypoints` pulls in OpenCV. Run the drift check once in an
environment that has it:

```python
from models import verify_joint_names
verify_joint_names()   # raises if the two declarations disagree
```

### Adjacency convention

`adjacency[k, source, target]`, normalized along the **source** axis, and the
aggregation is

```
out[target] = sum over source of x[source] * adjacency[k, source, target]
```

Summing the three partitions gives each target a set of weights adding to 1.
**Every model must sum over the source index.** Transposing it silently turns
the average into an unnormalized sum and swaps the centripetal and
centrifugal partitions. A regression test for this is one line: push a
constant field through the aggregation and check it comes back unchanged.

Partition 0 links joints at the same hop distance from the root, partition 1
points inward, partition 2 points outward.

## How missing joints are handled — `embed.py`

A joint MediaPipe never found is `(0, 0, 0)` with validity 0. After
`preprocess.dataset.normalize_clip` the origin is the shoulder midpoint, so
that hole sits exactly where a real joint resting on the chest would sit.
`neck` is literally defined as the shoulder midpoint, so it is `(0, 0, 0)`
with validity **1** in every frame of every clip.

**Never derive the mask from xyz.** `(xyz != 0).any()` marks `neck` as missing
in 100% of frames. Read channel 3.

Validity is not binary at the model input: `interpolate_clip` blends it, so a
phase next to a hole arrives with a fractional weight. After the resampling
fix, that fraction means *the xyz is a real observation carried over from the
nearest observed frame*, not *a partial hole*. The two states therefore get
two mechanisms:

| question | answer | mechanism |
|---|---|---|
| is this position real? | binary | **hard gate** on `validity > 0`, swapping in a per-joint learned token |
| how much do I trust it? | continuous | validity rides along as one of the 10 input channels |

Using the fractional value as the gate instead would shrink genuine motion
between two observed frames to a fraction of itself.

`SkeletonFeatures` builds 10 channels: xyz (3), frame-to-frame motion (3), the
vector to the BFS parent (3), and validity (1). Motion and bone are gated by
the hard mask of **both** endpoints, otherwise a hand reappearing produces a
velocity spike out of the origin.

`masked_global_pool` divides by the observed count rather than `T * J`. A plain
mean would hand the classifier a feature scaled down by however much of the
clip was a hole, so the same gloss would land at a different magnitude
depending on whether MediaPipe caught the hand.

### A diagnostic you should run

Adding validity opens a shortcut. If some glosses systematically lose a hand,
a model can classify by *what went missing when* instead of by handshape.
Train one baseline on the validity channel alone, with xyz zeroed. Chance on
422 classes is 0.24%; if that baseline clears roughly 5%, the mask leaks and
your headline number is partly fiction.

## Checkpoints — `classifiers.py`

No architecture lives here. Models are rebuilt through `build_model`, so the
graph used at inference cannot drift from the one used in training.

```python
from models import save_checkpoint, load_models, predict_gloss, pick_device

save_checkpoint(
    "checkpoints/vsl422_CTR-GCN_best.pth", model, "CTR-GCN",
    num_classes=422, is_leg=False, max_frames=64, epoch=37, val_acc=0.61,
)

device = pick_device()
loaded = load_models("checkpoints", num_classes=422, device=device)
result = predict_gloss(clip, loaded, glosses, device)   # clip: (T, J, 4)
```

A checkpoint records `model_name`, `num_classes`, `is_leg`, `max_frames`, and
the state dict; the loader rebuilds from that and refuses a class-count
mismatch instead of reshaping silently. `find_checkpoints` matches files named
`*_<MODEL NAME>_best.pth`.

`predict_gloss` averages softmax probabilities across the loaded models.
Picking the most confident model, which an earlier version did, only ever
rewards whichever head is worst calibrated. Per-model top-1 is still returned
under `predictions` so disagreement stays visible.

## Suggested training recipe

Not implemented here, but this is what the architectures were sized for.

- `CrossEntropyLoss` with `label_smoothing=0.1`. With 422 skewed classes this
  is not optional.
- SGD, `lr=0.1`, `momentum=0.9`, `nesterov=True`, `weight_decay=4e-4`, with
  batch norm and biases excluded from decay. Five warmup epochs, then cosine
  to zero over about 70.
- Gradient clipping at norm 1.0, and a weight EMA at decay 0.999.
- Report macro F1 next to accuracy. Accuracy alone on 422 skewed classes tells
  you very little.
- Order of work: `MS-TCN` for the floor, then `ST-GCN` for a real graph
  baseline, then tune `CTR-GCN`. Add `AAGCN` or `Transformer` only for an
  ensemble.

## Known gaps

- `preprocess.dataset.build_datasets` returns train and test only. There is no
  validation split, so there is nothing to select a checkpoint on except the
  test set, which inflates whatever you report.
- `preprocess.keypoints.to_model_input` still produces `(80, 228)` with three
  channels. The inference entry points, `infer.py` and
  `preprocess/camera.py`, therefore break at runtime against these models.
  Inference needs the training pipeline instead: `normalize_clip`,
  `drop_legs`, `interpolate_clip` to `T`. The normalization in
  `keypoints.normalize_body` and `keypoints.normalize_hands` is a different
  coordinate system and is not interchangeable.
- `preprocess/camera.py` asks for `names=("TCN",)`. That model is gone; the
  current name is `"MS-TCN"`.
- The `missing` token slice for `neck` is a dead parameter, since its validity
  is never 0. Harmless.
- No multi-stream ensemble (joint / bone / motion as separate towers) and no
  mask-aware graph convolution. Both were judged not worth the complexity
  until the per-joint hole rates say otherwise, because MediaPipe drops a hand
  as all 21 joints at once, leaving no neighbour to renormalize against.
