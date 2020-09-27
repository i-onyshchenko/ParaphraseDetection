import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, input_size=768, output_size=768):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        """

        :param inputs: list (shape (2,)) of tensors (batch_size, seq_len, embedding_size)
        :return: tensor of shape (batch_size,)
        """
        sentences1 = torch.mean(inputs[0], dim=1)
        sentences2 = torch.mean(inputs[1], dim=1)
        x1 = self.fc(sentences1)
        x1 = F.relu(x1)
        x2 = self.fc(sentences2)
        x2 = F.relu(x2)
        dist = F.cosine_similarity(x1, x2)

        return dist
