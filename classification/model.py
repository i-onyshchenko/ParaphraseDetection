from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from heads import ClassificationHead
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.base_model = AutoModel.from_pretrained("bert-base-uncased")
        self.classification_head = ClassificationHead(input_size=768, output_size=768)
        for param in self.base_model.parameters():
            param.requires_grad = False

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, inputs):
        """

        :param inputs: list of shape (2, batch_size)
        :return: tensor of shape (batch_size, 1)
        """
        tokens1 = self.tokenizer(inputs[0], truncation=True, padding=True, max_length=512, return_tensors="pt")
        tokens2 = self.tokenizer(inputs[1], truncation=True, padding=True, max_length=512, return_tensors="pt")
        embeddings1 = self.base_model(**tokens1)[0]
        embeddings2 = self.base_model(**tokens2)[0]

        logits = self.classification_head([embeddings1, embeddings2])

        return logits


if __name__ == "__main__":
    pass
