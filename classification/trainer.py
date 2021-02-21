# from transformers import Trainer, glue_convert_examples_to_features
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AdamW
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
# import tensorflow_datasets as tfds
import numpy as np
import random
import time
import datetime
from tqdm import tqdm

from model import Model
from nlp import load_dataset
from evaluation_utils import evaluate_classification, evaluate_embeddings
from util.utils import get_triplets

torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(666)
random.seed(666)


class MyTrainer:
    def __init__(self, model, dataset_name="mrpc", batch_size=12, epochs=30, epoch_size=80):
        self.model = model
        self.siam = True
        self.use_triplet = True
        if self.use_triplet:
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=0.5)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_size = epoch_size
        self.pretrain_head = False
        self.init_epochs = 10
        self.device = torch.device("cuda")

        self.model.to(self.device)

        # data = tfds.load('glue/mrpc')
        # self.train_dataset = glue_convert_examples_to_features(data['train'], self.tokenizer, max_length=128, task='mrpc')
        # self.train_dataset = self.train_dataset.shuffle(100).batch(32).repeat(2)
        # print(type(self.train_dataset))

        if self.dataset_name == "mrpc":
            self.train_dataset = load_dataset("glue", "mrpc", split="train")
            self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            self.test_dataset = load_dataset("csv", data_files={"test": "/home/ihor/University/DiplomaProject/Program/datasets/MRPC/msr_paraphrase_test.txt"},
                                             skip_rows=1, delimiter='\t', quote_char=False, column_names=['label', 'idx1', 'idx2', 'sentence1', 'sentence2'], split="test")
        elif self.dataset_name == "qqp":
            self.train_dataset = load_dataset("glue", "qqp", split="train")
            self.train_dataset = self.train_dataset.map(
                lambda examples: {'sentence1': examples['question1'], 'sentence2': examples['question2']}, batched=True)
            self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                                 shuffle=True)

            self.test_dataset = load_dataset("csv", data_files={"test": "/home/ihor/University/DiplomaProject/Program/datasets/Quora/dev.tsv"},
                                             skip_rows=1, delimiter='\t', quote_char=False, split="test")
            self.test_dataset = self.test_dataset.map(
                lambda examples: {'sentence1': examples['question1'], 'sentence2': examples['question2'], 'label': examples['is_duplicate']}, batched=True)
        else:
            raise Exception("Unsupported dataset!")
        self.nrof_train_samples = len(self.train_dataset["label"])

        print("Samples in train_set: {}".format(self.nrof_train_samples))
        self.nrof_test_samples = len(self.test_dataset["label"])
        print("Samples in test_set: {}".format(self.nrof_test_samples))

        # for logits of size (batch_size, nrof_classes)
        # self.criterion = nn.CrossEntropyLoss()

        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.01},
        #     {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # # print(self.model.named_parameters())
        # self.optimizer = AdamW(optimizer_grouped_parameters, lr=0.001)
        print("Nrof parameters: {}".format(len(list(self.model.parameters()))))

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01, nesterov=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    # def compute_loss(self, inputs):
    #     labels = inputs.pop("labels")
    #     outputs = self.model(**inputs)
    #     logits = outputs[0]
    #     return F.softmax(logits, labels)

    def train(self):
        # self.pretrain_head = False
        # self.init_epochs = 10
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            self.model.train()
            if self.pretrain_head:
                self.finetune_mode(self.init_epochs, epoch)

            for i, batch in enumerate(self.train_data_loader):
                start_time = time.time()
                # get the inputs; data is a list of [inputs, labels]

                sentences1 = batch["sentence1"]
                sentences2 = batch["sentence2"]

                inputs = [sentences1, sentences2]

                if self.siam:
                    _, embeddings1, embeddings2 = self.model.forward(inputs)
                else:
                    logits, _, _ = self.model.forward(inputs)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                if self.siam:
                    if self.use_triplet:
                        labels = list(batch['label'])
                        if labels.count(1) > 0:
                            anchor, positive, negative = get_triplets(embeddings1, embeddings2, labels)
                            loss = self.triplet_loss(anchor, positive, negative)
                        else:
                            i -= 1
                            continue
                    else:
                        # labels[i] \in {1, -1}
                        labels = torch.as_tensor(batch['label'] * 2 - 1, dtype=torch.float, device=self.device)
                        # loss = nn.CosineEmbeddingLoss(embeddings1, embeddings2, labels).to(self.device)
                        loss = F.cosine_embedding_loss(embeddings1, embeddings2, labels, margin=0.5)
                else:
                    # labels[i] in {1, 0}
                    labels = torch.as_tensor(batch['label'], dtype=torch.float, device=self.device)
                    loss = F.binary_cross_entropy(logits, labels)
                # loss = self.criterion(logits, labels).to(self.device)
                loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    # print statistics
                    print("Epoch: {}, {}/{}, time: {:.2f}, loss: {:.6f}".format(epoch + 1, i + 1, self.epoch_size, time.time() - start_time, loss.item()))

                if i + 1 == self.epoch_size:
                    break

            self.scheduler.step()
            self.evaluate(CLS=not self.siam, evaluate_softmax=False)

        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = "/home/ihor/University/DiplomaProject/Program/models/" + date_time + ".pt"
        torch.save(self.model, model_path)

    def finetune_mode(self, init_epochs, epoch):
        if epoch < init_epochs:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            # self.model.base_model.eval()
        elif epoch == init_epochs:
            for param in self.model.base_model.parameters():
                param.requires_grad = True
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01,
                                       nesterov=True)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.model.base_model.eval()

    def evaluate(self, CLS=True, **kwargs):
        if CLS:
            self.evaluate_cls(evaluate_softmax=kwargs.get("evaluate_softmax", False))
        else:
            self.evaluate_embeddings()

    def evaluate_cls(self, evaluate_softmax=False):
        """
        It is used for evaluation with binary output
        :return:
        """
        print('-' * 80)
        print("Evaluation...")

        sentences1 = self.test_dataset["sentence1"]
        sentences2 = self.test_dataset["sentence2"]

        if evaluate_softmax:
            logits = np.zeros((self.nrof_test_samples, 2))
        else:
            logits = np.zeros(self.nrof_test_samples)

        self.batch_size = 256
        steps = self.nrof_test_samples // self.batch_size

        # include last small batch if necessary
        if steps * self.batch_size != self.nrof_test_samples:
            steps += 1

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(steps)):
                inputs = [sentences1[i * self.batch_size:(i + 1) * self.batch_size],
                          sentences2[i * self.batch_size:(i + 1) * self.batch_size]]
                batch_logits, _, _ = self.model.to(self.device).forward(inputs)
                logits[i * self.batch_size:(i + 1) * self.batch_size] = batch_logits.cpu()

        labels = np.array(self.test_dataset["label"])

        if evaluate_softmax:
            predictions = np.argmax(logits, axis=1)
        else:
            predictions = logits
        tpr, fpr, accuracy, f1, best_thresholds = evaluate_classification(predictions, labels, nrof_folds=10)

        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('F1: %2.5f+-%2.5f' % (np.mean(f1), np.std(f1)))
        print('Best threshold: %2.5f+-%2.5f' % (np.mean(best_thresholds), np.std(best_thresholds)))

    def evaluate_embeddings(self):
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

        batch_size = 256
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

        tpr, fpr, accuracy, f1, best_thresholds = evaluate_embeddings(embeddings1, embeddings2, labels, nrof_folds=10,
                                                             distance_metric=1,
                                                             subtract_mean=False)

        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('F1: %2.5f+-%2.5f' % (np.mean(f1), np.std(f1)))
        print('Best threshold: %2.5f+-%2.5f' % (np.mean(best_thresholds), np.std(best_thresholds)))
        # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))


if __name__ == "__main__":
    model = Model()
    trainer = MyTrainer(model)
    trainer.train()

    # model = torch.load("/home/ihor/University/DiplomaProject/Program/models/20210120-232709.pt")
    # trainer = MyTrainer(model)
    # trainer.evaluate()
