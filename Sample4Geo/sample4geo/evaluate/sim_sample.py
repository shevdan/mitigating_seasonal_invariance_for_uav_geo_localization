"""
Similarity-based hard negative mining.

After each epoch, computes feature similarity between all training samples
and builds a dict of nearest neighbors in embedding space. Used to construct
batches where items are visually similar but from different locations.
"""

import gc

import torch
from tqdm import tqdm

from ..trainer import predict


def calc_sim(config, model, reference_dataloader, query_dataloader,
             neighbour_range=128, step_size=1000, cleanup=True):
    """Compute similarity-based neighbor dict from model embeddings.

    Args:
        reference_dataloader: Gallery (satellite) dataloader.
        query_dataloader: Query (drone) dataloader.
        neighbour_range: Number of nearest neighbors to keep.

    Returns:
        (r1_score, nearest_dict) where nearest_dict maps each query label
        to a list of reference labels sorted by embedding similarity.
    """
    print("\nExtract Features (sim_sample):")
    reference_features, reference_labels = predict(config, model, reference_dataloader)
    query_features, query_labels = predict(config, model, query_dataloader)

    Q = len(query_features)
    steps = Q // step_size + 1

    similarity = []
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())

    similarity = torch.cat(similarity, dim=0)

    # Top-k most similar references per query
    topk_scores, topk_ids = torch.topk(similarity, k=min(neighbour_range + 1, similarity.shape[1]), dim=1)

    topk_references = []
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i, :]])
    topk_references = torch.stack(topk_references, dim=0)

    # Mask out ground truth matches (same location = positive, not a hard negative)
    mask = topk_references != query_labels.unsqueeze(1)
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()

    nearest_dict = {}
    for i in range(len(topk_references)):
        nearest = topk_references[i][mask[i]][:neighbour_range]
        nearest_dict[query_labels[i].item()] = list(nearest)

    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()

    return nearest_dict
