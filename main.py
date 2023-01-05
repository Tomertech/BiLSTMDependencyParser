import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from model import MSTDependencyParser
from utils import get_sentences
import argparse


def exp(epochs, lr, batch_size, test, save, train_size):

    train_short_path = 'data/train_short.labeled'
    train_sample_path = 'data/train_short_5.labeled'
    train_path = 'data/train.labeled'
    test_path = 'data/test.labeled'
    test_path = 'data/test.labeled'
    comp_path = 'data/comp.unlabeled'
    model_path = 'models/model.pt'
    output_path = 'output.labeled'

    if train_size == "sample":
        train_path = train_sample_path
    elif train_size == "short":
        train_path = train_short_path
    elif train_size == "full":
        train_path = train_path

    train_sentences = get_sentences(train_path)

    model = MSTDependencyParser(sentences=train_sentences)

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"epochs: {epochs}, learning rate: {lr}, batch size: {batch_size}, test: {test}, save: {save}, train_size: {train_size}\n")

    model.train_model(epochs=epochs, lr=lr, batch_size=batch_size)
    train_uas = model.get_predictions(train_sentences)
    print(f"\ntrain mean UAS score: {train_uas}\n\n")

    if save:
        print(f"\nSaving model: {model_path}\n")
        torch.save(model, model_path)

    if test:
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        test_train_sentences = get_sentences(test_path)
        test_uas = model.get_predictions(test_train_sentences)
        print(f"\ntest mean UAS score: {test_uas}\n\n")


def run_exp():

    for i, lr in enumerate([5e-3, 1e-3, 5e-4, 1e-4, 5e-5]):
        print(f"\n\n############################### Start exp {i} with lr: {lr}###############################\n\n")
        exp(epochs=40, lr=lr, batch_size=32, test=True, save=False, train_size="full")
        print(f"\n\n############################### Finish exp {i} with lr: {lr}###############################\n\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='BiLSTM Dependency Parser')
    parser.add_argument('-e', '--epochs', dest='e', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=2e-3)
    parser.add_argument('-bs', '--batch_size', dest='bs', type=int, default=32)
    parser.add_argument('-t', '--test', dest='test', action='store_true')
    parser.add_argument('-s', '--save', dest='save', action='store_true')
    parser.add_argument('-ts', '--train_size', dest='ts', type=str, default='full')
    parser.add_argument('-me', '--multi_exp', dest='me', action='store_true')
    args = parser.parse_args()

    if args.me:
        run_exp()
    else:
        exp(args.e, args.lr, args.bs, args.test, args.save, args.ts)
