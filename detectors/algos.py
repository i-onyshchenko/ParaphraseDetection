import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm


def mean_phrase(batch_bag1, batch_bag2, threshold, **kwargs):
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


def pairs_matcher(batch_bag1, batch_bag2, **kwargs):
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
    # threshold = kwargs.get("threshold", 0.57)
    thresholds = kwargs.get("thresholds", [0.57])
    for dist in cos_dists:
        m, n = np.shape(dist)

        max_per_row = zip(range(m), np.argmax(dist, axis=1))
        max_per_col = zip(np.argmax(dist, axis=0), range(n))

        max_per_row = set([elem for elem in max_per_row if dist[elem[0], elem[1]] > thresholds[0]])
        max_per_col = set([elem for elem in max_per_col if dist[elem[0], elem[1]] > thresholds[0]])

        # preds.append(len(max_per_row.intersection(max_per_col)) > thresholds[1]*min(m, n))
        preds.append(len(max_per_row.intersection(max_per_col)) / min(m, n))

    return preds


def dependency_checker(batch_vect1, batch_vect2, dep_trees1=None, dep_trees2=None, **kwargs):
    """
    :param batch_vect1: batch of vectors
    :param batch_vect2: batch of vectors
    :param threshold:
    :param dep_trees1:
    :param dep_trees2
    :return: [1 if is paraphrase, else 0 for each entry]
    """
    # threshold = kwargs.get("threshold", 0.57)
    thresholds = kwargs.get("thresholds", [0.65, 0.5, 0.1])
    preds = []
    pos = {"PROPN", "NOUN", "VERB", "ADJ", "ADV"}
    dep = {"ROOT", "nsubj", "nsubjpass", "dobj", "pobj", "conj", "nmod", "obj", "iobj", "agent", "attr", "csubj", "csubjpass"}
    for i in range(len(batch_vect1)):
        sent1 = batch_vect1[i]
        sent2 = batch_vect2[i]
        tree1 = dep_trees1[i]
        tree2 = dep_trees2[i]

        # cos_dists = cosine_similarity(sent1, sent2)
        # neighbours1 = [[token.head.i] + [child.i for child in list(token.children)] for token in tree1 if token.dep_ != 'punct']
        neighbours1 = [[token.i] + [child.i for child in list(token.lefts)] + [child.i for child in list(token.rights)] for token in tree1 if token.dep_ != "punct"]
        neighbours2 = [[token.i] + [child.i for child in list(token.lefts)] + [child.i for child in list(token.rights)] for token in tree2 if token.dep_ != "punct"]
        # neighbours1 = [[token.i, token.head.i] + [child.i for child in list(token.subtree)] for token in tree1 if token.dep_ != "punct"]
        # neighbours2 = [[token.i, token.head.i] + [child.i for child in list(token.subtree)] for token in tree2 if token.dep_ != "punct"]
        # neighbours1 = [[token.i] for token in tree1 if token.dep_ != "punct"]
        # neighbours2 = [[token.i] for token in tree2 if token.dep_ != "punct"]
        # context_bags1 = [np.mean(sent1[indexes].numpy(), axis=0) for indexes in neighbours1]
        # context_bags2 = [np.mean(sent2[indexes].numpy(), axis=0) for indexes in neighbours2]
        context_bags1 = [np.mean(sent1[indexes].numpy(), axis=0) for indexes in neighbours1]
        context_bags2 = [np.mean(sent2[indexes].numpy(), axis=0) for indexes in neighbours2]

        m = len(context_bags1)
        n = len(context_bags2)

        # context_comparison = np.array([[pairs_matcher([context_bags1[i]], [context_bags2[j]], thresholds=thresholds[:1])[0] for j in range(n)] for i in range(m)])
        context_comparison = cosine_similarity(context_bags1, context_bags2)
        max_per_row = zip(range(m), np.argmax(context_comparison, axis=1))
        max_per_col = zip(np.argmax(context_comparison, axis=0), range(n))

        max_per_row = set([elem for elem in max_per_row if context_comparison[elem[0], elem[1]] > thresholds[0]])
        max_per_col = set([elem for elem in max_per_col if context_comparison[elem[0], elem[1]] > thresholds[0]])

        preds.append(len(max_per_row.intersection(max_per_col)) / max(m, n))
        # similarity = pairs_matcher([context_bags1], [context_bags2], threshold=threshold)
        #
        # preds += similarity

    return preds


def svm_bert(cls1, cls2, labels, test_cls1, test_cls2, test_labels):
    diff = [abs(cls1[i] - cls2[i]) for i in range(len(cls1))]
    prod = [cls1[i] * cls2[i] for i in range(len(cls1))]
    X = [np.concatenate((cls1[i].numpy(), cls2[i].numpy(), diff[i].numpy(), prod[i].numpy())) for i in range(len(cls1))]
    # X = [np.concatenate((cls1[i].numpy(), cls2[i].numpy())) for i in range(len(cls1))]
    clf = svm.SVC()
    clf.fit(X, labels)

    test_diff = [abs(test_cls1[i] - test_cls2[i]) for i in range(len(test_cls1))]
    test_prod = [test_cls1[i] * test_cls2[i] for i in range(len(test_cls1))]
    test_X = [np.concatenate((test_cls1[i].numpy(), test_cls2[i].numpy(), test_diff[i].numpy(), test_prod[i].numpy())) for i in range(len(test_cls1))]
    # test_X = [np.concatenate((test_cls1[i].numpy(), test_cls2[i].numpy())) for i in range(len(test_cls1))]

    preds = clf.predict(test_X)

    return preds

DETECTORS = {
    "mean_phrase": mean_phrase,
    "pairs_matcher": pairs_matcher,
    "dependency_checker": dependency_checker,
    "svm_bert": svm_bert,
}
