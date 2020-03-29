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
                'bert-base-uncased': 0.95,
                'openai-gpt': 0.87,
                'gpt2': 0.9985
             }


class ParaphraseDetectionModel:
    def __init__(self):
        pass

    def selectWords(self, tokenizer, sents):
        """
        :param tokenizer:
        :param sents:
        :return: indexes of valid words and tokenized sentence
        """
        nlp = spacy.load('en_core_web_sm')
        mod_sents = [nlp(sent) for sent in sents]

        tokenized_sents = [[tokenizer.encode(word.text, add_special_tokens=True)[1:-1] for word in sent] for sent in mod_sents]
        concat_tokenized_sents = [torch.tensor([[101] + sum(sent, []) + [102]]) for sent in tokenized_sents]

        indexes = []
        for (i, sent) in enumerate(tokenized_sents):
            index = []
            offset = 0
            for j in range(len(sent)):
                size = len(sent[j])
                if mod_sents[i][j].dep_ != 'punct':
                    index.append(offset + size - 1)
                    offset += size
                else:
                    offset += 1

            indexes.append(index)

        return indexes, concat_tokenized_sents

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
            print("Sentences were successfully tokenized!")

            selected_indexes1, tokenized_sents1 = self.selectWords(tokenizer, sents1)
            selected_indexes2, tokenized_sents2 = self.selectWords(tokenizer, sents2)

            with torch.no_grad():
                embeddings1 = [model(tok)[0][0] for tok in tokenized_sents1]
                embeddings2 = [model(tok)[0][0] for tok in tokenized_sents2]
                print("Embeddings were successfully created!")

                # selecting only embeddings that match words
                embeddings1 = [embeddings1[i][1:-1][selected_indexes1[i]] for i in range(len(embeddings1))]
                embeddings2 = [embeddings2[i][1:-1][selected_indexes2[i]] for i in range(len(embeddings2))]

                report = self.eval_model(DETECTORS['pairs_matcher'], embeddings1, embeddings2, labels, pretrained_weights)
                if verbose:
                    report_file.write("Model {}\n".format(pretrained_weights))
                    report_file.write(report)
                    report_file.write("\n\n")
                print(report)
                print("Model {} was successfully evaluated!\n".format(pretrained_weights))

        if verbose:
            report_file.close()
        print("Time elapsed: ", time.clock() - t_start)

    def eval_model(self, detector, tokens1, tokens2, labels, model_name):
        predictions = detector(tokens1, tokens2, THRESHOLDS.get(model_name, 0.9))
        return metrics.classification_report(labels, predictions)


if __name__ == "__main__":
    model = ParaphraseDetectionModel()
    data_set = MRPCDataSet(test_filename='datasets/MRPC/msr_paraphrase_test.txt')
    test_set = data_set.test_dataset
    if test_set is not None:
        model.evaluate(test_set.get_pairs, test_set.get_labels, verbose=True)
