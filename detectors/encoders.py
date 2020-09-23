import torch
from transformers import *
from sklearn import metrics
import numpy as np
import spacy
from spacy import displacy
from tqdm import tqdm, trange

from datasets.mrpc_dataset import MRPCDataSet
from datasets.quora_dataset import QuoraDataSet
from detectors.algos import DETECTORS

import os
import datetime
import time

import matplotlib.pyplot as plt

from itertools import product

# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`
#          Model          | Tokenizer          | Pretrained weights shortcut

MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
         ]

THRESHOLDS = {
                'bert-base-uncased': 0.90,
                'openai-gpt': 0.87,
                'gpt2': 0.9985,
                'xlnet-base-cased': 0.2
             }


class ParaphraseDetectionModel:
    def __init__(self):
        self.thresholds = np.linspace(0.0, 1.0, 101)

    def selectWordsBERT(self, tokenizer, sents):
        """
        primary for BERT
        :param tokenizer:
        :param sents:
        :return: indexes of valid words, tokenized sentence and dependency tree
        """
        nlp = spacy.load('en_core_web_sm')
        mod_sents = [nlp(sent) for sent in sents]

        # for token in mod_sents[0]:
        #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #           token.shape_, token.is_alpha, token.is_stop)

        # displacy.serve(mod_sents[0], style="dep")

        # neighbours1 = [[token.i, token.head.i] + [child.i for child in list(token.children) if child.dep_ != 'punct'] for token in mod_sents[0] if
        #                token.dep_ != 'punct']

        tokenized_sents = [[tokenizer.encode(word.text, add_special_tokens=True)[1:-1] for word in sent] for sent in mod_sents]
        concat_tokenized_sents = [torch.tensor([[101] + sum(sent, []) + [102]]) for sent in tokenized_sents]
        # concat_tokenized_sents = [torch.tensor([sum(sent, [])]) for sent in tokenized_sents]
        # print(tokenized_sents[0])
        # print(tokenizer.decode(tokenized_sents[0][0]))
        # print(tokenizer.decode(concat_tokenized_sents[0][0]))

        indexes = []
        for sent in tokenized_sents:
            index = []
            # to compensate first special token
            offset = 1
            for j in range(len(sent)):
                size = len(sent[j])
                # throw away punctuations
                # if mod_sents[i][j].dep_ != 'punct':
                #     index.append(offset + size - 1)
                #     offset += size
                # else:
                #     offset += 1
                index.append(offset + size - 1)
                offset += size

            indexes.append(index)

        # print(concat_tokenized_sents[0][0][indexes[0]])
        # for i in neighbours1:
        #     print(i)
        #     for elem in i:
        #         print(tokenizer.decode(tokenized_sents[0][elem]))
        # displacy.serve(mod_sents[0], style="dep")

        return indexes, concat_tokenized_sents, mod_sents

    def selectWordsRoBERTa(self, tokenizer, sents):
        """
        primary for BERT
        :param tokenizer:
        :param sents:
        :return: indexes of valid words, tokenized sentence and dependency tree
        """
        nlp = spacy.load('en_core_web_sm')
        mod_sents = [nlp(sent) for sent in sents]

        # for token in mod_sents[0]:
        #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #           token.shape_, token.is_alpha, token.is_stop)

        # displacy.serve(mod_sents[0], style="dep")

        # neighbours1 = [[token.i, token.head.i] + [child.i for child in list(token.children) if child.dep_ != 'punct'] for token in mod_sents[0] if
        #                token.dep_ != 'punct']


        tokenized_sents = [[tokenizer.encode(word.text, add_special_tokens=False) for word in sent] for sent in mod_sents]
        concat_tokenized_sents = [torch.tensor([[0] + sum(sent, []) + [2]]) for sent in tokenized_sents]
        # concat_tokenized_sents = [torch.tensor([sum(sent, [])]) for sent in tokenized_sents]
        # print(tokenized_sents[0])
        # print(tokenizer.decode(tokenized_sents[0][0]))
        # print(tokenizer.decode(concat_tokenized_sents[0][0]))

        indexes = []
        for (i, sent) in enumerate(tokenized_sents):
            index = []
            # to compensate first special token
            offset = 1
            for j in range(len(sent)):
                size = len(sent[j])
                # throw away punctuations
                # if mod_sents[i][j].dep_ != 'punct':
                #     index.append(offset + size - 1)
                #     offset += size
                # else:
                #     offset += 1
                index.append(offset + size - 1)
                offset += size

            indexes.append(index)

        # print(concat_tokenized_sents[0][0][indexes[0]])
        # for i in neighbours1:
        #     print(i)
        #     for elem in i:
        #         print(tokenizer.decode(tokenized_sents[0][elem]))
        # displacy.serve(mod_sents[0], style="dep")

        return indexes, concat_tokenized_sents, mod_sents

    def selectWordsXLNet(self, tokenizer, sents):
        """
        primary for XLNet
        :param tokenizer:
        :param sents:
        :return: indexes of valid words, tokenized sentence and dependency tree
        """
        # space - 17
        nlp = spacy.load('en_core_web_sm')
        mod_sents = [nlp(sent) for sent in sents]

        tokenized_sents = [[tokenizer.encode(word.text, add_special_tokens=False) for word in sent] for sent in
                           mod_sents]

        # print(tokenized_sents[0])
        concat_tokenized_sents = [torch.tensor([sum(sent, [])]) for sent in tokenized_sents]
        # print(tokenizer.decode(concat_tokenized_sents[0][0]))
        # print(tokenizer.decode([17]))
        indexes = []
        for (i, sent) in enumerate(tokenized_sents):
            index = []
            offset = 1
            for j in range(len(sent)):
                size = len(sent[j])
                # throw away punctuations
                # if mod_sents[i][j].dep_ != 'punct':
                #     index.append(offset + size - 1)
                #     offset += size
                # else:
                #     offset += 1
                index.append(offset + size - 1)
                offset += size

            indexes.append(index)

        return indexes, concat_tokenized_sents, mod_sents

    def evaluate(self, pairs=None, labels=None, test_pairs=None, test_labels=None, verbose=False):
        # Let's encode some text in a sequence of hidden-states using each model:
        t_start = time.clock()
        unziped = list(zip(*pairs))
        sents1, sents2 = unziped[0], unziped[1]
        if verbose:
            report_file = open(os.path.join('logs', list(DETECTORS.keys())[1] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt'), mode="w")
        for model_class, tokenizer_class, pretrained_weights in MODELS[:1]:
            # Load pretrained model/tokenizer
            tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            model = model_class.from_pretrained(pretrained_weights)

            # Encode text
            # tokenized_sents1 = [torch.tensor([tokenizer.encode(text=sent, add_special_tokens=True, add_space_before_punct_symbol=True)]) for sent in sents1]
            # tokenized_sents2 = [torch.tensor([tokenizer.encode(text=sent, add_special_tokens=True, add_space_before_punct_symbol=True)]) for sent in sents2]
            # print(tokenized_sents1[0][0])
            # print(tokenized_sents2[0][0])
            # print("Sentences were successfully tokenized!")

            print("Parsing first sentences...")
            selected_indexes1, tokenized_sents1, dep_trees1 = self.selectWordsBERT(tokenizer, sents1)
            print("Parsing second sentences...")
            selected_indexes2, tokenized_sents2, dep_trees2 = self.selectWordsBERT(tokenizer, sents2)

            with torch.no_grad():
                print("Generating first embeddings...")
                embeddings1 = [model(tok)[0][0] for tok in tqdm(tokenized_sents1)]
                print("Generating second embeddings...")
                embeddings2 = [model(tok)[0][0] for tok in tqdm(tokenized_sents2)]
                print("Embeddings were successfully created!")

                if test_pairs is not None:
                    cls1 = [embed.mean(dim=0) for embed in embeddings1]
                    cls2 = [embed.mean(dim=0) for embed in embeddings2]

                    test_unziped = list(zip(*test_pairs))
                    test_sents1, test_sents2 = test_unziped[0], test_unziped[1]

                    _, test_tokenized_sents1, _ = self.selectWordsBERT(tokenizer, test_sents1)
                    _, test_tokenized_sents2, _ = self.selectWordsBERT(tokenizer, test_sents2)

                    test_embeddings1 = [model(tok)[0][0] for tok in test_tokenized_sents1]
                    test_embeddings2 = [model(tok)[0][0] for tok in test_tokenized_sents2]

                    test_cls1 = [embed.mean(dim=0) for embed in test_embeddings1]
                    test_cls2 = [embed.mean(dim=0) for embed in test_embeddings2]

                    preds = DETECTORS["svm_bert"](cls1, cls2, labels, test_cls1, test_cls2, test_labels)
                    report = metrics.classification_report(test_labels, preds, output_dict=False)
                else:
                    # selecting only embeddings that match words
                    embeddings1 = [embeddings1[i][selected_indexes1[i]] for i in range(len(embeddings1))]
                    embeddings2 = [embeddings2[i][selected_indexes2[i]] for i in range(len(embeddings2))]

                    print("Time for embeddings: ", time.clock() - t_start)

                    report = self.eval_model(DETECTORS['dependency_checker'], embeddings1, embeddings2, labels, pretrained_weights, dep_trees1=dep_trees1, dep_trees2=dep_trees2)
                    # report = self.find_thresholds(2, DETECTORS['dependency_checker'], embeddings1, embeddings2, labels,
                    #                      pretrained_weights, dep_trees1=dep_trees1, dep_trees2=dep_trees2)

                if verbose:
                    report_file.write("Model {}\n".format(pretrained_weights))
                    report_file.write(report)
                    report_file.write("\n\n")
                print(report)
                print("Model {} was successfully evaluated!\n".format(pretrained_weights))

        if verbose:
            report_file.close()
        print("Time elapsed: ", time.clock() - t_start)

    def find_thresholds(self, num_thresholds, detector, tokens1, tokens2, labels, model_name, **kwargs):
        thresholds = product(*np.repeat(self.thresholds[np.newaxis, ...], num_thresholds-1, axis=0))
        best_acc = 0
        best_acc_thresholds = None
        aux_f1 = None
        best_f1 = 0
        best_f1_thresholds = None
        aux_acc = None
        best_sum = 0
        sum_acc = None
        sum_f1 = None
        sum_thresholds = None
        for th in thresholds:
            print(th)
            predictions = detector(tokens1, tokens2, thresholds=th, dep_trees1=kwargs.get("dep_trees1", None),
                                   dep_trees2=kwargs.get("dep_trees2", None))
            for final_th in self.thresholds:
                thresholded = [1 if pred > final_th else 0 for pred in predictions]
                report = metrics.classification_report(labels, thresholded, output_dict=True)

                if report['accuracy'] > best_acc:
                    best_acc = report['accuracy']
                    best_acc_thresholds = list(th) + [final_th]
                    aux_f1 = report['1']['f1-score']
                if report['1']['f1-score'] > best_f1:
                    best_f1 = report['1']['f1-score']
                    best_f1_thresholds = list(th) + [final_th]
                    aux_acc = report['accuracy']
                if report['accuracy'] + report['1']['f1-score'] > best_sum:
                    best_sum = report['accuracy'] + report['1']['f1-score']
                    sum_acc = report['accuracy']
                    sum_f1 = report['1']['f1-score']
                    sum_thresholds = list(th) + [final_th]

        print("Max F1-score = {:.3} (accuracy = {:.3}) at thresholds {}".format(best_f1, aux_acc, best_f1_thresholds))
        print("Max Accuracy = {:.3} (f1-score = {:.3}) at thresholds {}".format(best_acc, aux_f1, best_acc_thresholds))
        print("Best combo: Accuracy =  {:.3}, f1-score = {:.3} at thresholds {}".format(sum_acc, sum_f1, sum_thresholds))

        return report

    def eval_model(self, detector, tokens1, tokens2, labels, model_name, **kwargs):
        f1_scores = []
        acurracies = []
        predictions = detector(tokens1, tokens2, threshold=0.57, dep_trees1=kwargs.get("dep_trees1", None), dep_trees2=kwargs.get("dep_trees2", None))

        self.histogram(predictions, labels, show=False)
        # for th in self.thresholds:
        for th in [0.53]:
            # predictions = detector(tokens1, tokens2, THRESHOLDS.get(model_name, 0.9), dep_trees1=kwargs.get("dep_trees1", None), dep_trees2=kwargs.get("dep_trees2", None))
            thresholded = [1 if pred > th else 0 for pred in predictions]
            report = metrics.classification_report(labels, thresholded, output_dict=True)
            print(report)
            f1_scores.append(report['1']['f1-score'])
            acurracies.append(report['accuracy'])


        # calculate f1-score and accuracy for all-1 classification
        report = metrics.classification_report(labels, np.ones(len(labels)), output_dict=True)
        dummy_f1_score = report['1']['f1-score']
        dummy_acc = report['accuracy']

        print(f1_scores)
        print(acurracies)
        max_f1_index = np.argmax(f1_scores)
        max_acc_index = np.argmax(acurracies)
        print("Max F1-score = {:.3} (accuracy = {:.3}) at threshold {:.2}".format(f1_scores[max_f1_index], acurracies[max_f1_index], self.thresholds[max_f1_index]))
        print("Max Accuracy = {:.3} (f1-score = {:.3}) at threshold {:.2}".format(acurracies[max_acc_index], f1_scores[max_acc_index], self.thresholds[max_acc_index]))

        fig, axs = plt.subplots(2)
        axs[0].set_ylim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.0])
        axs[0].set_title("F1")
        axs[1].set_title("Accuracy")
        axs[0].plot(self.thresholds, f1_scores, label="Mine")
        axs[0].plot(self.thresholds, [dummy_f1_score] * len(self.thresholds), label="Dummy")
        axs[0].legend()
        axs[1].plot(self.thresholds, acurracies, label="Mine")
        axs[1].plot(self.thresholds, [dummy_acc] * len(self.thresholds), label="Dummy")
        axs[1].legend()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return report

    def histogram(self, predictions, labels, **kwargs):
        pos = []
        neg = []

        for i in range(len(labels)):
            if labels[i] == 1:
                pos.append(predictions[i])
            else:
                neg.append(predictions[i])

        names = ["Pos", "Neg"]
        plt.hist([pos, neg], bins=100, label=names)

        # Plot formatting
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Predictions')

        show = kwargs.get("show", True)
        if show:
            plt.show()


if __name__ == "__main__":
    model = ParaphraseDetectionModel()
    data_set = MRPCDataSet(train_filename='datasets/MRPC/msr_paraphrase_train.txt', test_filename='datasets/MRPC/msr_paraphrase_test.txt')
    test_set = data_set.test_dataset
    # data_set = QuoraDataSet(test_filename="datasets/Quora/dev.tsv")
    # test_set = data_set.test_dataset
    if test_set is not None:
        # model.evaluate(train_set.get_pairs, train_set.get_labels, test_set.get_pairs, test_set.get_labels, verbose=False)
        model.evaluate(test_set.get_pairs, test_set.get_labels, verbose=False)
