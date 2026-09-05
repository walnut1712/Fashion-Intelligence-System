"""Task 4 training components, shared between resolutions.

The sampler, the loss and the augmentation pipeline lived only inside notebook
06, which was fine while 60x80 was the only resolution and the notebook was the
only caller. Moving Task 4 to the 120x160 catalogue needs a script - the run is
long enough that it should survive a closed laptop - and a script cannot copy
these out of a notebook without creating the second definition CLAUDE.md warns
about.

So they live here, and nothing in this module declares a network:
``ImprovedEncoder`` is imported from ``src/visual_search/search_engine.py``,
which stays the single definition the service loads.

Notebook 06's own CELL 7 predates this module. Its cells are executed and are
the 60x80 record, so they are left alone rather than rewritten to import; the
definitions here are copied from them verbatim and the tests assert the two
agree.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.synthetic_backgrounds import (
    TRAINING_DEGRADATIONS,
    composite,
    degrade,
    simulate_ingestion,
)

__all__ = [
    "PKSampler",
    "batch_hard_triplet_loss",
    "augment_batch",
    "WildDataset",
    "embedding_spread",
    "channel_statistics",
]


class PKSampler(torch.utils.data.Sampler):
    """P classes x K images per batch, avoiding two photos of one product.

    A triplet loss needs several images of the same class in a batch or it has
    no positive to mine. Drawing two photographs of one product would give it a
    trivially easy positive instead, which is why the product id is tracked.
    """

    def __init__(self, labels, product_ids, p=16, k=8, batches_per_epoch=250,
                 seed=42):
        self.p, self.k, self.batches_per_epoch = p, k, batches_per_epoch
        self.rng = np.random.default_rng(seed)
        self.by_class = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_class[label].append(index)
        self.classes = [c for c, items in self.by_class.items() if len(items) >= k]
        self.product_ids = np.asarray(product_ids)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch = []
            for cls in self.rng.choice(self.classes, min(self.p, len(self.classes)),
                                       replace=False):
                candidates = self.by_class[cls]
                picked, seen = [], set()
                for index in self.rng.permutation(candidates):
                    if self.product_ids[index] in seen:
                        continue
                    picked.append(index); seen.add(self.product_ids[index])
                    if len(picked) == self.k:
                        break
                while len(picked) < self.k:
                    picked.append(int(self.rng.choice(candidates)))
                batch.extend(picked)
            yield batch


def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """Hardest positive against hardest negative, within the batch."""
    distances = torch.cdist(embeddings, embeddings, p=2)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask, negative_mask = same & ~eye, ~same
    hardest_positive = (distances * positive_mask).max(dim=1).values
    hardest_negative = (distances + (~negative_mask).float() * 1e6).min(dim=1).values
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    loss = F.relu(hardest_positive - hardest_negative + margin)
    return loss[valid].mean() if valid.any() else loss.sum() * 0.0


def augment_batch(x, flip=True, jitter=0.15):
    """Flip and photometric jitter, applied to the already-normalised batch."""
    if flip:
        do = torch.rand(x.size(0), device=x.device) < 0.5
        x = torch.where(do.view(-1, 1, 1, 1), torch.flip(x, dims=[3]), x)
    if jitter > 0:
        brightness = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2 - 1) * jitter
        contrast = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2 - 1) * jitter
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * contrast + mean * brightness
    return x


class WildDataset(Dataset):
    """Catalogue frames put through the distribution the service actually sees.

    Three stages, in the order a photograph acquires them: swap the backdrop,
    let a camera degrade it, then let the ingestion pipeline try to segment it.
    ``probability`` and ``strength`` are ramped by the caller rather than fixed,
    because starting either at full strength collapsed the run that notebook 06
    section 6 records.

    Resolution is taken from the arrays, never assumed.
    """

    def __init__(self, images, masks, backgrounds, positions, types, colours,
                 mean, std, scale_range=(0.55, 1.00), augment=True, seed=42):
        self.images, self.masks, self.backgrounds = images, masks, backgrounds
        self.positions = np.asarray(positions)
        self.types = np.asarray(types)
        self.colours = np.asarray(colours)
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.scale_range = scale_range
        self.augment = augment
        self.generator = np.random.default_rng(seed)

        self.probability = 0.0        # backdrop swap
        self.strength = 0.0           # camera degradation
        self.p_segment = None         # None -> the measured 62/38 default

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        position = self.positions[index]
        image = np.asarray(self.images[position])

        if self.augment and self.generator.random() < self.probability:
            background = self.backgrounds[
                self.generator.integers(len(self.backgrounds))]
            image = composite(image, np.asarray(self.masks[position]), background,
                              self.generator, scale_range=self.scale_range)

            # Only degraded frames go through ingestion. A clean catalogue image
            # is already what nobg would produce from it, and segmenting it
            # would teach a crop production never applies to a flat lay.
            if self.strength > 0:
                image = degrade(image, self.generator, TRAINING_DEGRADATIONS,
                                strength=self.strength)
                kwargs = ({} if self.p_segment is None
                          else {"p_segment": self.p_segment})
                image = simulate_ingestion(image, self.generator, **kwargs)

        tensor = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0)
        tensor = (tensor - self.mean) / self.std
        return tensor, int(self.types[index]), int(self.colours[index])


@torch.no_grad()
def embedding_spread(model, images, positions, mean, std, device, sample=512,
                     seed=0):
    """Mean pairwise distance between embeddings. Near zero means collapse."""
    model.eval()
    chosen = np.sort(np.random.default_rng(seed).choice(positions, sample,
                                                        replace=False))
    # 512 frames, read in one go: ~29 MB at 120x160, which is affordable where
    # the full catalogue is not.
    chunk = np.asarray(images[chosen], dtype=np.float32) / 255.0
    tensor = torch.from_numpy(chunk.transpose(0, 3, 1, 2))
    tensor = ((tensor - torch.as_tensor(mean).view(1, 3, 1, 1))
              / torch.as_tensor(std).view(1, 3, 1, 1)).to(device)
    vectors = model.embed(tensor)
    return float(torch.cdist(vectors, vectors).mean())


def channel_statistics(images, positions, sample=4000, seed=42, chunk=256):
    """Per-channel mean and std, computed on the TRAINING rows only.

    Fitting these over the whole gallery would leak held-out pixels into the
    normalisation every model uses.

    Accumulated in float64, in chunks. The obvious one-liner - stack the sample
    and call ``.mean()`` on it - is wrong at this size and wrong *silently*: at
    120x160 a 4,000-image sample is 76.8 million values per channel, and a
    float32 running sum passes its 24-bit mantissa around 16.7 million, after
    which adding 0.85 changes nothing. It returned [0.2185, 0.2185, 0.2185] for
    a catalogue of white-background product shots whose true mean is 0.84, and
    every model normalised with it would have trained on badly centred inputs.
    The 60x80 arm sat just under the limit and looked correct, which is how a
    bug like this survives a smaller test.
    """
    chosen = np.sort(np.random.default_rng(seed).choice(
        positions, min(sample, len(positions)), replace=False))

    total = np.zeros(3, dtype=np.float64)
    total_square = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(chosen), chunk):
        block = np.asarray(images[chosen[start:start + chunk]],
                           dtype=np.float64) / 255.0
        total += block.sum(axis=(0, 1, 2))
        total_square += (block ** 2).sum(axis=(0, 1, 2))
        count += block.shape[0] * block.shape[1] * block.shape[2]

    mean = total / count
    variance = np.maximum(total_square / count - mean ** 2, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
