import torch
import torch.nn as nn
import torch.nn.functional as F


# use if you wanna evaluating embeddings using some distance, e.g., cosine distance (similarity)
class DummyHead(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(DummyHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        """

        :param inputs: list (shape (2,)) of tensors (batch_size, seq_len, embedding_size)
        :return: tensor of shape (batch_size,)
        """
        sentences1 = torch.mean(inputs[0], dim=1)
        sentences2 = torch.mean(inputs[1], dim=1)
        x1 = self.fc(sentences1)
        x1 = torch.tanh(x1)
        x2 = self.fc(sentences2)
        x2 = torch.tanh(x2)
        # 1 - paraphrase, 0 - otherwise
        score = F.cosine_similarity(x1, x2)
        score = torch.relu(score)

        return score, x1, x2


class GLUEHead(nn.Module):
    def __init__(self, input_size=768*4, output_size=1):
        super(GLUEHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        original_x1 = torch.mean(inputs[0], dim=1)
        original_x2 = torch.mean(inputs[1], dim=1)
        abs_diff_x = torch.abs(original_x1 - original_x2)
        product_x = original_x1*original_x2

        x = torch.cat((original_x1, original_x2, abs_diff_x, product_x), dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        logits = F.sigmoid(x)

        return logits.squeeze()
