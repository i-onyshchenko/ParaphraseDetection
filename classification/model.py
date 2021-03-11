from transformers import AutoTokenizer, AutoModel, AutoModelForPreTraining
import torch.nn as nn
from heads import SiamHead, SemiSiamHead, CLSHead
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
        self.base_model = AutoModel.from_pretrained("bert-base-cased-finetuned-mrpc")
        # self.base_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # self.base_model = AutoModel.from_pretrained("bert-base-cased")
        # self.base_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        # self.base_model = AutoModel.from_pretrained("albert-base-v2")
        # self.base_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        # self.base_model = AutoModel.from_pretrained("roberta-base")
        # self.base_tokenizer = AutoTokenizer.from_pretrained("DeBERTa/deberta-base")
        # self.base_model = AutoModel.from_pretrained("DeBERTa/deberta-base")
        # self.base_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        # self.base_model = AutoModel.from_pretrained("microsoft/deberta-base")
        # print(self.base_model.config)
        # self.base_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")
        # self.base_model = AutoModel.from_pretrained("nlpaueb/legal-bert-small-uncased")
        self.base_embedding_size = 768
        # self.classification_head = GLUEHead()
        # self.classification_head = GLUEHead(input_size=self.base_embedding_size)
        self.siam = True
        self.semi_siam = False
        self.CLS = False
        self.aggregation_type = "CLS"
        if self.siam:
            self.last_layer_size = 768
            self.classification_head = SiamHead(input_size=self.base_embedding_size, output_size=self.last_layer_size,
                                                aggregation_type=self.aggregation_type)
        elif self.semi_siam:
            self.last_layer_size = 1
            self.classification_head = SemiSiamHead(input_size=self.base_embedding_size, output_size=self.last_layer_size,
                                                    aggregation_type=self.aggregation_type)
        elif self.CLS:
            self.last_layer_size = 2
            self.classification_head = CLSHead(input_size=self.base_embedding_size, output_size=self.last_layer_size,
                                               aggregation_type=self.aggregation_type)
        else:
            raise Exception("Not supported head!")

        self.device = "cuda"

        for param in self.base_model.parameters():
            param.requires_grad = True

    @property
    def tokenizer(self):
        return self.base_tokenizer

    def forward(self, inputs):
        """

        :param inputs: list of shape (2, batch_size)
        :param siam: flag to use siamese network
        :return: tensor of shape (batch_size, 1)
        """
        if self.siam:
            tokens1 = self.base_tokenizer(inputs[0], truncation=True, padding=True, max_length=128, return_tensors="pt")
            tokens1 = {key: value.to(self.device) for key, value in tokens1.items()}
            tokens2 = self.base_tokenizer(inputs[1], truncation=True, padding=True, max_length=128, return_tensors="pt")
            tokens2 = {key: value.to(self.device) for key, value in tokens2.items()}
            embeddings1 = self.base_model.to(self.device)(**tokens1)[0]
            embeddings2 = self.base_model.to(self.device)(**tokens2)[0]

            attentions = [tokens1["attention_mask"], tokens2["attention_mask"]]
            _, embedds1, embedds2 = self.classification_head.to(self.device)([embeddings1, embeddings2], attentions=attentions)
            return _, embedds1, embedds2
        elif self.semi_siam:
            swap_inputs = False

            tokens1 = self.base_tokenizer(inputs[0], truncation=True, padding=True, max_length=128, return_tensors="pt")
            tokens1 = {key: value.to(self.device) for key, value in tokens1.items()}
            tokens2 = self.base_tokenizer(inputs[1], truncation=True, padding=True, max_length=128, return_tensors="pt")
            tokens2 = {key: value.to(self.device) for key, value in tokens2.items()}
            embeddings1 = self.base_model.to(self.device)(**tokens1)[0]
            embeddings2 = self.base_model.to(self.device)(**tokens2)[0]

            attentions = [tokens1["attention_mask"], tokens2["attention_mask"]]
            logits, _, _ = self.classification_head.to(self.device)([embeddings1, embeddings2],
                                                                    attentions=attentions)

            if swap_inputs:
                tokens_pair2 = self.base_tokenizer(inputs[1], inputs[0], truncation=True, padding=True, max_length=512,
                                                   return_tensors="pt")
                tokens_pair2 = {key: value.to(self.device) for key, value in tokens_pair2.items()}
                embeddings2 = self.base_model.to(self.device)(**tokens_pair2)[0]
                logits2, _, _ = self.classification_head.to(self.device)(embeddings2, attentions=tokens_pair2["attention_mask"])

                return (logits + logits2) / 2, None, None

            return logits, None, None
        elif self.CLS:
            swap_inputs = True

            tokens_pair = self.base_tokenizer(inputs[0], inputs[1], truncation=True, padding=True, max_length=512,
                                              return_tensors="pt")
            tokens_pair = {key: value.to(self.device) for key, value in tokens_pair.items()}
            # print(tokens_pair["attention_mask"])
            embeddings = self.base_model.to(self.device)(**tokens_pair)[0]
            logits, _, _ = self.classification_head.to(self.device)(embeddings, attentions=tokens_pair["attention_mask"])

            if swap_inputs:
                tokens_pair2 = self.base_tokenizer(inputs[1], inputs[0], truncation=True, padding=True, max_length=512,
                                                  return_tensors="pt")
                tokens_pair2 = {key: value.to(self.device) for key, value in tokens_pair2.items()}
                embeddings2 = self.base_model.to(self.device)(**tokens_pair2)[0]
                logits2, _, _ = self.classification_head.to(self.device)(embeddings2, attentions=tokens_pair2["attention_mask"])

                return (logits + logits2) / 2, None, None

            # print(tokens_pair["attention_mask"][0])
            # print(embeddings[0].size())
            # print(embeddings[0][tokens_pair["attention_mask"][0] == 1].size())

            return logits, None, None
        else:
            raise Exception("Not specified head!")

    @property
    def embed_size(self):
        return self.base_embedding_size

    @property
    def output_size(self):
        return self.last_layer_size


if __name__ == "__main__":
    pass
