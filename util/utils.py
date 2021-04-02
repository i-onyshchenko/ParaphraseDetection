import torch
import numpy as np


def masked_aggregation(batch, mask, func="CLS"):
    # print("Batch shape: {}".format(batch.shape))
    # print("Mask shape: {}".format(mask.shape))
    batch_zero = batch * mask.unsqueeze(dim=-1)

    if func == "mean":
        # skip CLS token
        lens = mask.sum(dim=1) - 1
        # print(lens.shape)
        batch_res = batch_zero[:, 1:].sum(dim=1) / lens.unsqueeze(dim=-1)
        # print(batch_res.shape)
    elif func == "max":
        # batch_res = batch_zero.max(dim=1)
        batch_res, _ = torch.max(batch_zero[:, 1:], dim=1, keepdim=False)
    elif func == "CLS":
        batch_res = batch[:, 0]
    else:
        print("Unsupported aggregation! Using CLS.")
        batch_res = batch[:, 0]

    return batch_res


def get_triplets(embeddings1, embeddings2, labels):
    '''
    :param embeddings1: torch.tensor of shape (bs, emb_size)
    :param embeddings2: torch.tensor of shape (bs, emb_size)
    :param labels: list of labels
    :return: torch.tensor of shape (bs, emb_size), torch.tensor of shape (bs, emb_size), torch.tensor of shape (bs, emb_size)
    '''

    nrof_labels = len(labels)
    labels = np.array(labels)
    positive_indices = np.where(labels == 1)[0]
    if positive_indices.size == 0:
        raise Exception("No anchors!")

    # for each positive pair let's create 4 triplets:
    # (pos1, pos2, neg1), (pos1, pos2, neg2), (pos2, pos1, neg1), (pos2, pos1, neg2)
    anchors, positive, negative = [], [], []
    for pos_index in positive_indices:
        neg_index = (pos_index + 1) % nrof_labels
        anchors += [embeddings1[pos_index], embeddings1[pos_index], embeddings2[pos_index], embeddings2[pos_index]]
        positive += [embeddings2[pos_index], embeddings2[pos_index], embeddings1[pos_index], embeddings1[pos_index]]
        negative += [embeddings1[neg_index], embeddings2[neg_index], embeddings1[neg_index], embeddings2[neg_index]]

    return torch.stack(anchors), torch.stack(positive), torch.stack(negative)




