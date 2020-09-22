from transformers import Trainer
from torch.nn import functional as F

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        return F.softmax(logits, labels)


if __name__ == "__main__":
    pass