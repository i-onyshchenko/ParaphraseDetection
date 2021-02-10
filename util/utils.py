import torch


def masked_aggregation(batch, mask, func="mean"):
    # print("Batch shape: {}".format(batch.shape))
    # print("Mask shape: {}".format(mask.shape))
    batch_zero = batch * mask.unsqueeze(dim=-1)

    if func == "mean":
        lens = mask.sum(dim=1)
        # print(lens.shape)
        batch_res = batch_zero.sum(dim=1) / lens.unsqueeze(dim=-1)
        # print(batch_res.shape)
    elif func == "max":
        # batch_res = batch_zero.max(dim=1)
        batch_res, _ = torch.max(batch_zero, dim=1, keepdim=True)
    else:
        print("Unsupported aggregation! Using mean.")
        lens = mask.sum(dim=1)
        batch_res = batch_zero.sum(dim=1) / lens.unsqueeze(dim=-1)

    return batch_res