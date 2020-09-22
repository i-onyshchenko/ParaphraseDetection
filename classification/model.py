from transformers import AutoTokenizer, AutoModel
import torch


class Model(torch.nn.Module):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
        model = AutoModel.from_pretrained("bert-base-uncased")

    def forward(self, inputs):
        return inputs


if __name__ == "__main__":
    pass
