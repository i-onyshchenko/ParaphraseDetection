from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        for param in self.model.parameters():
            param.requires_grad = False

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, inputs):
        return inputs


if __name__ == "__main__":
    pass
