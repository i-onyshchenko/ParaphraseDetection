from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import easygui


def demoHuggingFace():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

    classes = ["not paraphrase", "is paraphrase"]

    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

    paraphrase = tokenizer.encode_plus(sequence_0, sequence_2, return_tensors="pt")
    not_paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, return_tensors="pt")

    paraphrase_classification_logits = model(**paraphrase)[0]
    not_paraphrase_classification_logits = model(**not_paraphrase)[0]

    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
    not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

    print("Should be paraphrase")
    for i in range(len(classes)):
        print(f"{classes[i]}: {round(paraphrase_results[i] * 100)}%")

    print("\nShould not be paraphrase")
    for i in range(len(classes)):
        print(f"{classes[i]}: {round(not_paraphrase_results[i] * 100)}%")


def demoMasters():
    # model = torch.load("/home/ihor/University/DiplomaProject/Program/models/bert_base_cls_20210401-132024.pt")
    model = torch.load("/home/ihor/University/DiplomaProject/Program/models/roberta_base_cls_20210401-180654.pt")
    # model = torch.load("/home/ihor/University/DiplomaProject/Program/models/roberta_base_finetuned_semi_siam_20210402-190739.pt")
    # model = torch.load("/home/ihor/University/DiplomaProject/Program/models/albert_base_v2_cls20210401-155540.pt")
    device = torch.device("cuda")
    model.to(device)

    # sequence_0 = "The company HuggingFace is based in New York City"
    # sequence_1 = "HuggingFace's headquarters are situated in Manhattan"
    # sequence_2 = "Apples are especially bad for your health"

    sequence_0 = easygui.enterbox("Write down the first sentence: ")
    sequence_1 = easygui.enterbox("Write down the second sentence: ")
    # sequence_2 = easygui.enterbox("Write down an arbitrary sentence: ")

    # # bert
    # threshold = 0.67940
    # roberta
    threshold = 0.32560
    # # roberta finetuned semi-siam
    # threshold = 0.50600
    # # albert
    # threshold = 0.22330

    model.eval()
    with torch.no_grad():
        logits_par, _, _ = model.forward([sequence_0, sequence_1])
        # logits_non_par, _, _ = model.forward([sequence_0, sequence_2])

    print("Threshold for identifying a paraphrase: {:.5f}".format(threshold))
    print("Sentences:\n1. {}\n2. {}\n--> Semantic similarity: {:.5f}".format(sequence_0, sequence_1, logits_par.cpu().numpy()))
    # print("Sentences:\n1. {}\n2. {}\n--> Semantic similarity: {:.5f}".format(sequence_0, sequence_2, logits_non_par.cpu().numpy()))


if __name__ == "__main__":
    # demoHuggingFace()
    demoMasters()
