from transformers import Trainer, glue_convert_examples_to_features
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AdamW
# import tensorflow_datasets as tfds
import numpy as np

from model import Model
from nlp import load_dataset


class MyTrainer():
    def __init__(self, model, batch_size=32, epochs=20, epoch_size=20):
        # super(MyTrainer, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_size = epoch_size

        # data = tfds.load('glue/mrpc')
        # self.train_dataset = glue_convert_examples_to_features(data['train'], self.tokenizer, max_length=128, task='mrpc')
        # self.train_dataset = self.train_dataset.shuffle(100).batch(32).repeat(2)
        # print(type(self.train_dataset))
        self.train_dataset = load_dataset("glue", "mrpc", split="train")
        self.nrof_samples = len(self.train_dataset["label"])

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
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            # TODO: add iterator/sampler
            for i in range(self.epoch_size):
                # get the inputs; data is a list of [inputs, labels]
                batch_indexes = np.random.choice(self.nrof_samples, self.batch_size, replace=False)
                sentences1 = [self.train_dataset["sentence1"][idx] for idx in batch_indexes]
                sentences2 = [self.train_dataset["sentence2"][idx] for idx in batch_indexes]
                inputs = [sentences1, sentences2]
                labels = torch.as_tensor([self.train_dataset["label"][idx] for idx in batch_indexes], dtype=torch.float)

                logits = self.model.forward(inputs)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                loss = F.binary_cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                print("Epoch: {}, {}/{}, loss: {}".format(epoch + 1, i + 1, self.epoch_size, loss.item()))

            self.evaluate()

    def evaluate(self):
        print('-'*80)
        print("Evaluation...")


if __name__ == "__main__":
    model = Model()
    trainer = MyTrainer(model)
    trainer.train()
    # train_dataset = load_dataset("glue", "mrpc", split="test")
    # print(train_dataset.column_names)
    # print(len(train_dataset['sentence2']))