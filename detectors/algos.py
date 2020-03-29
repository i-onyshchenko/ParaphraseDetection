import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def mean_phrase(batch_bag1, batch_bag2, threshold):
    """
    Compute Cosine distance between means of bags
    :param batch_bag1: list of tensors, size (batch_size, num_of_features)
    :param batch_bag2:
    :param threshold:
    :return: [1 if is paraphrase, else 0 for each entry]
    """
    batch_avg1 = [entry.mean(axis=0) for entry in batch_bag1]
    batch_avg2 = [entry.mean(axis=0) for entry in batch_bag2]
    cos_dists = [cosine_similarity([np.array(batch_avg1[i])], [np.array(batch_avg2[i])]) for i in range(len(batch_avg1))]
    return [1 if dist > threshold else 0 for dist in cos_dists]


def pairs_matcher(batch_bag1, batch_bag2, threshold):
    """
    Computes pair-wise similarity between words in bags
    :param batch_bag1: list of tensors, size (batch_size, num_of_features)
    :param batch_bag2:
    :param threshold:
    :return: [1 if is paraphrase, else 0 for each entry]
    """
    batch_bag1 = [bag for bag in batch_bag1]
    batch_bag2 = [bag for bag in batch_bag2]

    cos_dists = [cosine_similarity(bag1, bag2) for (bag1, bag2) in zip(batch_bag1, batch_bag2)]

    preds = []
    # counts number of close pairs
    # for dist in cos_dists:
    #     m, n = np.shape(dist)
    #
    #     max_per_row = np.max(dist, axis=1)
    #     max_per_col = np.max(dist, axis=0)
    #
    #     matches_in_row = np.count_nonzero(max_per_row > threshold)
    #     matches_in_col = np.count_nonzero(max_per_col > threshold)
    #
    #     preds.append(matches_in_row + matches_in_col > 0.6*(m+n))

    for dist in cos_dists:
        m, n = np.shape(dist)

        max_per_row = set(np.max(dist, axis=1))
        max_per_col = set(np.max(dist, axis=0))

        preds.append(len(max_per_row.intersection(max_per_col)) > 0.7*min(m, n))

    return preds


def dependency_checker(batch_sent1, batch_sent2, batch_vect1, batch_vect2, threshold):
    """
    :param batch_sent1: batch of sentences
    :param batch_sent2: batch of sentences
    :param batch_vect1: batch of vectors
    :param batch_vect2: batch of vectors
    :param threshold:
    :return: [1 if is paraphrase, else 0 for each entry]
    """
    pass


DETECTORS = {
    "mean_phrase": mean_phrase,
    "pairs_matcher": pairs_matcher,
    "dependency_checker": dependency_checker
}
