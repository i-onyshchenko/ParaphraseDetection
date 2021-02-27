import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import masked_aggregation


# use if you wanna evaluating embeddings using some distance, e.g., cosine distance (similarity)
class SiamHead(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(SiamHead, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        # self.batch_norm = nn.BatchNorm1d(256 * 4)
        self.aggregation_type = "CLS"

    def forward(self, inputs, attentions=None):
        """

        :param inputs: list (shape (2,)) of tensors (batch_size, seq_len, embedding_size)
        :param attentions: list (shape (2,)) of tensors (batch_size, seq_len)
        :return: tensor of shape (batch_size,)
        """

        sentences1 = masked_aggregation(inputs[0], attentions[0], self.aggregation_type)
        sentences2 = masked_aggregation(inputs[1], attentions[1], self.aggregation_type)
        # sentences1 = torch.mean(inputs[0], dim=1)
        # sentences2 = torch.mean(inputs[1], dim=1)

        # Branch for Sentence 1
        x1 = sentences1
        # x1 = self.fc1(sentences1)
        # x1 = torch.tanh(x1)
        # x1 = self.dropout(x1)
        #
        # x1 = self.fc2(x1)
        # x1 = torch.tanh(x1)
        # x1 = self.dropout(x1)
        #
        # x1 = self.fc3(x1)
        # x1 = torch.tanh(x1)
        # x1 = torch.relu(x1)
        # x1 = self.fc2(x1)
        # x1 = torch.tanh(x1)

        # Branch for Sentence 2
        x2 = sentences2
        # x2 = self.fc1(sentences2)
        # x2 = torch.tanh(x2)
        # x2 = self.dropout(x2)
        #
        # x2 = self.fc2(x2)
        # x2 = torch.tanh(x2)
        # x2 = self.dropout(x2)
        #
        # x2 = self.fc3(x2)
        # x2 = torch.tanh(x2)
        # x2 = torch.relu(x2)
        # x2 = self.fc2(x2)
        # x2 = torch.tanh(x2)

        # Comparison of embeddings
        # 1 - paraphrase, 0 - otherwise
        # score = F.cosine_similarity(x1, x2)
        # print(score)
        # to clip negative values
        # score = torch.relu(score)
        # print(score)

        return None, x1, x2


class SemiSiamHead(nn.Module):
    def __init__(self, input_size=768, output_size=2):
        super(SemiSiamHead, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size*4, self.output_size)
        # self.fc1 = nn.Linear(self.input_size*4, 512)
        # self.fc2 = nn.Linear(512, self.output_size)
        # self.fc3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        # self.batch_norm = nn.BatchNorm1d(256 * 4)
        self.aggregation_type = "CLS"

    def forward(self, inputs, attentions=None):
        """

        :param inputs: list (shape (2,)) of tensors (batch_size, seq_len, embedding_size)
        :param attentions: list (shape (2,)) of tensors (batch_size, seq_len)
        :return: tensor of shape (batch_size,)
        """

        sentences1 = masked_aggregation(inputs[0], attentions[0], self.aggregation_type)
        sentences2 = masked_aggregation(inputs[1], attentions[1], self.aggregation_type)

        # Branch for Sentence 1
        x1 = sentences1

        # Branch for Sentence 2
        x2 = sentences2

        x = torch.cat([x1, x2, torch.abs(x1-x2), x1*x2], dim=-1)
        x = self.fc(x)
        # x = self.fc1(x)
        # x = torch.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        if self.output_size == 1:
            x = torch.sigmoid(x)

        return x.squeeze(), None, None


class CLSHead(nn.Module):
    def __init__(self, input_size=768, output_size=2):
        super(CLSHead, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # self.fc1 = nn.Linear(input_size // 4, 256)
        # self.fc2 = nn.Linear(input_size // 4, 256)
        # self.fc3 = nn.Linear(input_size // 4, 256)
        # self.fc4 = nn.Linear(256*4, output_size)
        self.hidden_fc = nn.Linear(self.input_size, 1024)
        self.output_layer = nn.Linear(1024, self.output_size)
        self.dropout = nn.Dropout(p=0.5)
        # self.batch_norm1 = nn.BatchNorm1d(input_size)
        # self.batch_norm2 = nn.BatchNorm1d(1024)
        self.aggregation_type = "CLS"

    def forward(self, inputs, attentions=None):
        # original_x1 = torch.mean(inputs[0], dim=1)
        # # original_x1 = inputs[0][:, 0]
        # original_x2 = torch.mean(inputs[1], dim=1)
        # # original_x2 = inputs[1][:, 0]
        # abs_diff_x = torch.abs(original_x1 - original_x2)
        # product_x = original_x1*original_x2
        #
        # original_x1 = self.fc1(original_x1)
        # original_x2 = self.fc1(original_x2)
        # abs_diff_x = self.fc2(abs_diff_x)
        # product_x = self.fc3(product_x)
        #
        # x = torch.cat((original_x1, original_x2, abs_diff_x, product_x), dim=1)
        # # x = torch.cat((original_x1, original_x2), dim=1)
        #
        # # x = self.dropout(x)
        # # x = self.batch_norm(x)
        # x = self.fc4(x)
        # # logits = F.sigmoid(x)

        # x = torch.mean(inputs[:, :1], dim=1)
        # print(inputs[0])
        x = masked_aggregation(inputs, attentions, self.aggregation_type)
        # x = inputs[:, 0]
        # x = self.dropout(x)
        # x = torch.relu(x)
        # x = self.dropout(x)
        # x = self.batch_norm1(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        # x = self.dropout(x)
        # x = self.batch_norm2(x) # make it worse
        x = self.output_layer(x)

        if self.output_size == 1:
            x = torch.sigmoid(x)

        return x.squeeze(), None, None
