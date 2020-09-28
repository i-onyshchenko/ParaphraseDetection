from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from heads import ClassificationHead
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.base_model = AutoModel.from_pretrained("bert-base-uncased")
        self.last_layer_size = 768
        self.classification_head = ClassificationHead(input_size=768, output_size=self.last_layer_size)
        self.device = "cuda"
        for param in self.base_model.parameters():
            param.requires_grad = False

    @property
    def tokenizer(self):
        return self.base_tokenizer

    def forward(self, inputs):
        """

        :param inputs: list of shape (2, batch_size)
        :return: tensor of shape (batch_size, 1)
        """
        tokens1 = self.tokenizer(inputs[0], truncation=True, padding=True, max_length=512, return_tensors="pt")
        tokens1 = {key: value.to(self.device) for key, value in tokens1.items()}
        tokens2 = self.tokenizer(inputs[1], truncation=True, padding=True, max_length=512, return_tensors="pt")
        tokens2 = {key: value.to(self.device) for key, value in tokens2.items()}
        embeddings1 = self.base_model.to(self.device)(**tokens1)[0]
        embeddings2 = self.base_model.to(self.device)(**tokens2)[0]

        logits, embeddings1, embeddings2 = self.classification_head.to(self.device)([embeddings1, embeddings2])

        return logits, embeddings1, embeddings2

    @property
    def output_size(self):
        return self.last_layer_size


if __name__ == "__main__":
    pass
