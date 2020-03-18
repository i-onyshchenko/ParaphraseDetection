import torch
from transformers import *
from sklearn import metrics

from datasets.mrpc_dataset import MRPCDataSet
from detectors.algos import DETECTORS

import os
import datetime

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
                'bert-base-uncased': 0.9,
                'openai-gpt': 0.87,
                'gpt2': 0.9985
             }


class ParaphraseDetectionModel:
    def __init__(self):
        pass

    def evaluate(self, pairs=None, labels=None):
        # Let's encode some text in a sequence of hidden-states using each model:
        unziped = list(zip(*pairs))
        words1, words2 = unziped[0], unziped[1]
        report_file = open(os.path.join('logs', list(DETECTORS.keys())[0] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt'), mode="w")
        for model_class, tokenizer_class, pretrained_weights in MODELS[:1]:
            # Load pretrained model/tokenizer
            tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            model = model_class.from_pretrained(pretrained_weights)

            # Encode text
            tokenized_words1 = [torch.tensor([tokenizer.encode(text=word, add_special_tokens=True, add_space_before_punct_symbol=True)]) for word in words1]
            tokenized_words2 = [torch.tensor([tokenizer.encode(text=word, add_special_tokens=True, add_space_before_punct_symbol=True)]) for word in words2]
            print("Sentences were successfully tokenized!")

            with torch.no_grad():
                embeddings1 = [model(tok)[0][0] for tok in tokenized_words1]
                embeddings2 = [model(tok)[0][0] for tok in tokenized_words2]
                print("Embeddings were successfully created!")

                report = self.eval_model(DETECTORS['mean_phrase'], embeddings1, embeddings2, labels, pretrained_weights)
                report_file.write("Model {}\n".format(pretrained_weights))
                report_file.write(report)
                report_file.write("\n\n")
                print(report)
                print("Model {} was successfully evaluated!\n".format(pretrained_weights))

        report_file.close()

    def eval_model(self, detector, tokens1, tokens2, labels, model_name):
        predictions = [detector(tokens1[i], tokens2[i], THRESHOLDS.get(model_name, 0.9)) for i in range(len(labels))]
        # print("Labels:      ", labels)
        # print("Predictions: ", predictions)
        return metrics.classification_report(labels, predictions)


if __name__ == "__main__":
    model = ParaphraseDetectionModel()
    data_set = MRPCDataSet(test_filename='datasets/MRPC/msr_paraphrase_test.txt')
    test_set = data_set.test_dataset
    if test_set is not None:
        model.evaluate(test_set.get_pairs, test_set.get_labels)











