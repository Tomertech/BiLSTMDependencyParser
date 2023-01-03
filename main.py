import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from model import MSTDependencyParser
from utils import get_transformed_sentences, get_pre_trained_voacb, get_pos_vocab


def main():

    train_short_path = 'data/train_short.labeled'
    train_short_5_path = 'data/train_short_5.labeled'
    train_path = 'data/train.labeled'
    test_path = 'data/test.labeled'
    test_path = 'data/test.labeled'
    comp_path = 'data/comp.unlabeled'
    model_path = 'models/model.pt'
    output_path = 'output.labeled'

    pre_trained_vocab = get_pre_trained_voacb()
    pos_vocab = get_pos_vocab()
    train_transformed_sentences = get_transformed_sentences(train_path, pre_trained_vocab, pos_vocab)

    model = MSTDependencyParser(sentences=train_transformed_sentences, word_vocab=pre_trained_vocab, pos_vocab=pos_vocab)

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~Training~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    model.train_model(epochs=10, lr=1e-3, batch_size=10)
    train_uas = model.get_uas_corpus(train_transformed_sentences)
    print(f"\ntrain mean UAS score: {train_uas}\n\n")

    # torch.save(model, model_path)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    test_transformed_sentences = get_transformed_sentences(test_path, pre_trained_vocab, pos_vocab)
    test_uas = model.get_uas_corpus(test_transformed_sentences)
    print(f"\ntest mean UAS score: {test_uas}\n\n")


if __name__ == '__main__':
    main()
