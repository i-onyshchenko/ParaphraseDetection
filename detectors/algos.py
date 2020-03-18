def mean_phrase(bag1, bag2, threshold):
    """
    Compute Cosine distance between means of bags
    :param bag1:
    :param bag2:
    :param threshold:
    :return: 1 if is paraphrase, else 0
    """
    avg1 = bag1.mean(axis=0)
    avg2 = bag2.mean(axis=0)
    norm1 = avg1.pow(2).sum().sqrt()
    norm2 = avg2.pow(2).sum().sqrt()
    # dist = (avg1 - avg2).pow(2).sum().sqrt()
    cos_dist = avg1.dot(avg2) / norm1 / norm2
    # print("Distance: ", cos_dist)
    return 1 if cos_dist > threshold else 0


def pairs_matcher(bag1, bag2, threshold):
    """
    Computes pair-wise similarity between words in bags
    :param bag1:
    :param bag2:
    :param threshold:
    :return: 1 if is paraphrase, else 0
    """
    pass

DETECTORS = {
    "mean_phrase": mean_phrase,
}