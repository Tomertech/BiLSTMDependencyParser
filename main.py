import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from model import MSTDependencyParser
from utils import get_sentences


def main():

    train_short_path = 'data/train_short.labeled'
    train_short_5_path = 'data/train_short_5.labeled'
    train_path = 'data/train.labeled'
    test_path = 'data/test.labeled'
    test_path = 'data/test.labeled'
    comp_path = 'data/comp.unlabeled'
    model_path = 'models/model.pt'
    output_path = 'output.labeled'

    train_sentences = get_sentences(train_short_path)

    model = MSTDependencyParser(sentences=train_sentences)

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~Training~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    model.train_model(epochs=10, lr=5e-3, batch_size=10)
    train_uas = model.get_uas_corpus(train_sentences)
    print(f"\ntrain mean UAS score: {train_uas}\n\n")

    # torch.save(model, model_path)

    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    # test_train_sentences = get_sentences(test_path)
    # test_uas = model.get_uas_corpus(test_train_sentences)
    # print(f"\ntest mean UAS score: {test_uas}\n\n")


if __name__ == '__main__':
    main()
