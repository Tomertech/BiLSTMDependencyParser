import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from model import MSTDependencyParser
from utils import get_sentences
import argparse


def main(epochs, lr, batch_size):

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

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"epochs: {epochs}, learning rate: {lr}, batch size: {batch_size}\n")

    model.train_model(epochs=epochs, lr=lr, batch_size=batch_size)
    train_uas = model.get_uas_corpus(train_sentences)
    print(f"\ntrain mean UAS score: {train_uas}\n\n")

    # torch.save(model, model_path)
    #
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    # test_train_sentences = get_sentences(test_path)
    # test_uas = model.get_uas_corpus(test_train_sentences)
    # print(f"\ntest mean UAS score: {test_uas}\n\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='BiLSTM Dependency Parser')
    parser.add_argument('-e', '--epochs', dest='e', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=5e-3)
    parser.add_argument('-bs', '--batch_size', dest='bs', type=int, default=100)
    args = parser.parse_args()

    main(args.e, args.lr, args.bs)
