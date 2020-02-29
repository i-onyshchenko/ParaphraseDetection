import torch
from transformers import *
from mrpc_dataset import MRPCDataSet
from sklearn import metrics

# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

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


class ParaphraseDetectionModel:
    def __init__(self):
        pass

    def evaluate(self, pairs=None, labels=None):
        # Let's encode some text in a sequence of hidden-states using each model:
        unziped = list(zip(*pairs))
        words1, words2 = unziped[0], unziped[1]
        for model_class, tokenizer_class, pretrained_weights in MODELS[:1]:
            # Load pretrained model/tokenizer
            tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            model = model_class.from_pretrained(pretrained_weights)

            # Encode text
            tokenized_words1 = [torch.tensor([tokenizer.encode(text=word, add_special_tokens=True)]) for word in words1]
            tokenized_words2 = [torch.tensor([tokenizer.encode(text=word, add_special_tokens=True)]) for word in words2]

            with torch.no_grad():
                embeddings1 = [model(tok)[0] for tok in tokenized_words1]
                embeddings2 = [model(tok)[0] for tok in tokenized_words2]

            self.eval_model(embeddings1, embeddings2, labels)

    def eval_model(self, tokens1, tokens2, labels):
        ziped_tokens = zip(tokens1, tokens2)
        predictions = [self.detect(tok[0], tok[1]) for tok in ziped_tokens]
        print(metrics.classification_report(labels, predictions))

    def detect(self, bag1, bag2):
        return 0


if __name__ == "__main__":
    model = ParaphraseDetectionModel()
    data_set = MRPCDataSet(test_filename='MRPC/msr_paraphrase_test.txt')
    test_set = data_set.get_test_dataset
    if test_set is not None:
        model.evaluate(test_set.get_pairs, test_set.get_labels)