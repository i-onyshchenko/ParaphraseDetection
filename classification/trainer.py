from transformers import Trainer, glue_convert_examples_to_features
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AdamW
# import tensorflow_datasets as tfds
import numpy as np
import time
from tqdm import tqdm

from model import Model
from nlp import load_dataset
from evaluation_utils import evaluate_classification, evaluate_embeddings


class MyTrainer:
    def __init__(self, model, batch_size=256, epochs=50, epoch_size=10):
        # super(MyTrainer, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_size = epoch_size
        self.device = "cuda"

        # data = tfds.load('glue/mrpc')
        # self.train_dataset = glue_convert_examples_to_features(data['train'], self.tokenizer, max_length=128, task='mrpc')
        # self.train_dataset = self.train_dataset.shuffle(100).batch(32).repeat(2)
        # print(type(self.train_dataset))
        self.train_dataset = load_dataset("glue", "mrpc", split="train")
        self.nrof_train_samples = len(self.train_dataset["label"])

        self.test_dataset = load_dataset("csv", data_files={"test": "/home/ihor/University/DiplomaProject/Program/datasets/MRPC/msr_paraphrase_test.txt"},
                                         skip_rows=1, delimiter='\t', quote_char=False, column_names=['label', 'idx1', 'idx2', 'sentence1', 'sentence2'], split="test")
        self.nrof_test_samples = len(self.test_dataset["label"])

        self.criterion = nn.CrossEntropyLoss()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3)
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
                start_time = time.time()
                # get the inputs; data is a list of [inputs, labels]
                batch_indexes = np.random.choice(self.nrof_train_samples, self.batch_size, replace=False)
                sentences1 = [self.train_dataset["sentence1"][idx] for idx in batch_indexes]
                sentences2 = [self.train_dataset["sentence2"][idx] for idx in batch_indexes]
                inputs = [sentences1, sentences2]
                labels = torch.as_tensor([self.train_dataset["label"][idx] for idx in batch_indexes], dtype=torch.float, device=self.device)

                logits = self.model.to(self.device).forward(inputs)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                loss = F.binary_cross_entropy(logits, labels)
                # loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                print("Epoch: {}, {}/{}, time: {:.2f}, loss: {:.2f}".format(epoch + 1, i + 1, self.epoch_size, time.time() - start_time, loss.item()))

            self.evaluate()

    def evaluate(self):
        """
        It is used for evaluation with binary output
        :return:
        """
        print('-' * 80)
        print("Evaluation...")

        sentences1 = self.test_dataset["sentence1"]
        sentences2 = self.test_dataset["sentence2"]

        logits = np.zeros(self.nrof_test_samples)

        steps = self.nrof_test_samples // self.batch_size

        # include last small batch if necessary
        if steps * self.batch_size != self.nrof_test_samples:
            steps += 1

        with torch.no_grad():
            for i in tqdm(range(steps)):
                inputs = [sentences1[i * self.batch_size:(i + 1) * self.batch_size],
                          sentences2[i * self.batch_size:(i + 1) * self.batch_size]]
                batch_logits = self.model.to(self.device).forward(inputs)
                logits[i * self.batch_size:(i + 1) * self.batch_size] = batch_logits.cpu()

        labels = np.array(self.test_dataset["label"])

        tpr, fpr, accuracy, f1 = evaluate_classification(logits, labels, nrof_folds=10)

        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('F1: %2.5f+-%2.5f' % (np.mean(f1), np.std(f1)))


    def evaluate_embed(self):
        """
        It is used for classification with custom distance (cosine, euclid) between embeddings
        :return:
        """
        print('-'*80)
        print("Evaluation...")

        sentences1 = self.test_dataset["sentence1"]
        sentences2 = self.test_dataset["sentence2"]

        embeddings1 = np.zeros((self.nrof_test_samples, self.model.output_size))
        embeddings2 = np.zeros((self.nrof_test_samples, self.model.output_size))

        batch_size = 128
        steps = self.nrof_test_samples // batch_size

        # include last small batch if necessary
        if steps*batch_size != self.nrof_test_samples:
            steps += 1

        with torch.no_grad():
            for i in tqdm(range(steps)):
                inputs = [sentences1[i*batch_size:(i+1)*batch_size], sentences2[i*batch_size:(i+1)*batch_size]]
                _, sent1, sent2 = self.model.to(self.device).forward(inputs)
                embeddings1[i*batch_size:(i+1)*batch_size] = sent1.cpu()
                embeddings2[i*batch_size:(i+1)*batch_size] = sent2.cpu()

        labels = np.array(self.test_dataset["label"])

        tpr, fpr, accuracy, f1, val, val_std, far = evaluate_embeddings(embeddings1, embeddings2, labels, nrof_folds=10,
                                                             distance_metric=1,
                                                             subtract_mean=False)

        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('F1: %2.5f+-%2.5f' % (np.mean(f1), np.std(f1)))
        # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))


if __name__ == "__main__":
    model = Model()
    trainer = MyTrainer(model)
    trainer.train()
    # train_dataset = load_dataset("glue", "mrpc", split="test")
    # print(train_dataset.column_names)
    # print(len(train_dataset['sentence2']))