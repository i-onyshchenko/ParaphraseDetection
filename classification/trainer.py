from transformers import Trainer, glue_convert_examples_to_features
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AdamW
import tensorflow_datasets as tfds

from .model import Model


class MyTrainer(Trainer):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tokenizer = self.model.get_tokenizer()

        data = tfds.load('glue/mrpc')
        self.train_dataset = glue_convert_examples_to_features(data['train'], self.tokenizer, max_length=128, task='mrpc')
        self.train_dataset = self.train_dataset.shuffle(100).batch(32).repeat(2)
        print(type(self.train_dataset))

        self.criterion = nn.CrossEntropyLoss()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    # def compute_loss(self, inputs):
    #     labels = inputs.pop("labels")
    #     outputs = self.model(**inputs)
    #     logits = outputs[0]
    #     return F.softmax(logits, labels)

    def train(self):
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            # TODO: add iterator/sampler
            for i, data in enumerate(range(2), 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


if __name__ == "__main__":
    model = Model()
    trainer = Trainer(model)