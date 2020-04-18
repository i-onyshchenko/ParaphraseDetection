import torch
from transformers import *
from sklearn import metrics
import numpy as np
import spacy

from datasets.mrpc_dataset import MRPCDataSet
from detectors.algos import DETECTORS

import os
import datetime
import time

import matplotlib.pyplot as plt

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
        self.thresholds = np.linspace(0.0, 1.0, 100)

    def selectWordsBERT(self, tokenizer, sents):
        """
        primary for BERT
        :param tokenizer:
        :param sents:
        :return: indexes of valid words, tokenized sentence and dependency tree
        """
        nlp = spacy.load('en_core_web_sm')
        mod_sents = [nlp(sent) for sent in sents]

        tokenized_sents = [[tokenizer.encode(word.text, add_special_tokens=True)[1:-1] for word in sent] for sent in mod_sents]
        concat_tokenized_sents = [torch.tensor([[101] + sum(sent, []) + [102]]) for sent in tokenized_sents]
        # concat_tokenized_sents = [torch.tensor([sum(sent, [])]) for sent in tokenized_sents]

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
            offset = 0
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

    def evaluate(self, pairs=None, labels=None, verbose=False):
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
            print("Sentences were successfully tokenized!")

            selected_indexes1, tokenized_sents1, dep_trees1 = self.selectWordsBERT(tokenizer, sents1)
            selected_indexes2, tokenized_sents2, dep_trees2 = self.selectWordsBERT(tokenizer, sents2)

            with torch.no_grad():
                embeddings1 = [model(tok)[0][0] for tok in tokenized_sents1]
                embeddings2 = [model(tok)[0][0] for tok in tokenized_sents2]
                print("Embeddings were successfully created!")

                # selecting only embeddings that match words
                embeddings1 = [embeddings1[i][selected_indexes1[i]] for i in range(len(embeddings1))]
                embeddings2 = [embeddings2[i][selected_indexes2[i]] for i in range(len(embeddings2))]

                print("Time for embeddings: ", time.clock() - t_start)

                report = self.eval_model(DETECTORS['pairs_matcher'], embeddings1, embeddings2, labels, pretrained_weights, dep_trees1=dep_trees1, dep_trees2=dep_trees2)
                if verbose:
                    report_file.write("Model {}\n".format(pretrained_weights))
                    report_file.write(report)
                    report_file.write("\n\n")
                print(report)
                print("Model {} was successfully evaluated!\n".format(pretrained_weights))

        if verbose:
            report_file.close()
        print("Time elapsed: ", time.clock() - t_start)

    def eval_model(self, detector, tokens1, tokens2, labels, model_name, **kwargs):
        f1_scores = []
        acurracies = []
        for th in self.thresholds:
            # predictions = detector(tokens1, tokens2, THRESHOLDS.get(model_name, 0.9), dep_trees1=kwargs.get("dep_trees1", None), dep_trees2=kwargs.get("dep_trees2", None))
            predictions = detector(tokens1, tokens2, th, dep_trees1=kwargs.get("dep_trees1", None), dep_trees2=kwargs.get("dep_trees2", None))
            report = metrics.classification_report(labels, predictions, output_dict=True)
            f1_scores.append(report['1']['f1-score'])
            acurracies.append(report['accuracy'])

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
        axs[0].plot(self.thresholds, f1_scores)
        axs[1].plot(self.thresholds, acurracies)

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return report


if __name__ == "__main__":
    model = ParaphraseDetectionModel()
    data_set = MRPCDataSet(test_filename='datasets/MRPC/msr_paraphrase_test.txt')
    test_set = data_set.test_dataset
    if test_set is not None:
        model.evaluate(test_set.get_pairs, test_set.get_labels, verbose=False)
