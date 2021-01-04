from transformers import AutoTokenizer, AutoModel, AutoModelForPreTraining
import torch.nn as nn
from heads import CosineHead, GLUEHead
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
        self.base_model = AutoModel.from_pretrained("bert-base-cased-finetuned-mrpc")
        # self.base_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")
        # self.base_model = AutoModel.from_pretrained("nlpaueb/legal-bert-small-uncased")
        self.base_embedding_size = 768
        self.last_layer_size = 1
        # self.classification_head = GLUEHead()
        self.classification_head = GLUEHead(input_size=self.base_embedding_size)
        self.device = "cuda"
        i = 0
        for param in self.base_model.parameters():
            i += 1
            param.requires_grad = True
            # if i == 190:
            #     break

    @property
    def tokenizer(self):
        return self.base_tokenizer

    def forward(self, inputs):
        """

        :param inputs: list of shape (2, batch_size)
        :return: tensor of shape (batch_size, 1)
        """
        tokens_pair = self.base_tokenizer(inputs[0], inputs[1], truncation=True, padding=True, max_length=512, return_tensors="pt")
        tokens_pair = {key: value.to(self.device) for key, value in tokens_pair.items()}
        # tokens1 = self.base_tokenizer(inputs[0], truncation=True, padding=True, max_length=128, return_tensors="pt")
        # tokens1 = {key: value.to(self.device) for key, value in tokens1.items()}
        # tokens2 = self.base_tokenizer(inputs[1], truncation=True, padding=True, max_length=128, return_tensors="pt")
        # tokens2 = {key: value.to(self.device) for key, value in tokens2.items()}
        # embeddings1 = self.base_model.to(self.device)(**tokens1)[0]
        # embeddings2 = self.base_model.to(self.device)(**tokens2)[0]
        #
        # # use this line if head returns its own embeddings
        # # logits, embeddings1, embeddings2 = self.classification_head.to(self.device)([embeddings1, embeddings2])
        # logits = self.classification_head.to(self.device)([embeddings1, embeddings2])
        embeddings = self.base_model.to(self.device)(**tokens_pair)[0]
        # print(tokens_pair["attention_mask"][0])
        # print(embeddings[0].size())
        # print(embeddings[0][tokens_pair["attention_mask"][0] == 1].size())
        # exit()
        logits = self.classification_head.to(self.device)(embeddings)

        return logits

    @property
    def embed_size(self):
        return self.base_embedding_size

    @property
    def output_size(self):
        return self.last_layer_size


if __name__ == "__main__":
    pass
