import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from model import MSTDependencyParser
from utils import get_sentences, write_sentences, device
import argparse


def exp(epochs, lr, batch_size, test, save, train_size):

    train_short_path = 'data/train_short.labeled'
    train_sample_path = 'data/train_short_5.labeled'
    train_path = 'data/train.labeled'
    test_path = 'data/test.labeled'
    test_path = 'data/test.labeled'
    train_and_test_concat = 'data/train_and_test_concat.labeled'
    comp_path = 'data/comp.unlabeled'
    model_path = 'models/model.pt'
    output_path = 'output.labeled'

    if train_size == "sample":
        train_path = train_sample_path
    elif train_size == "short":
        train_path = train_short_path
    elif train_size == "full":
        train_path = train_path
    elif train_size == "concat":
        train_path = train_and_test_concat

    train_sentences = get_sentences(train_path)
    test_sentences = get_sentences(test_path) if test else None

    model = MSTDependencyParser(sentences=train_sentences)

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"epochs: {epochs}, learning rate: {lr}, batch size: {batch_size}, test: {test}, save: {save}, "
          f"train_size: {train_size}, torch seed: {0}, vocab: GloVe.6B.100\n")

    model.train_model(epochs=epochs, lr=lr, batch_size=batch_size, test_sentences=test_sentences)
    train_uas = model.eval_model(train_sentences)
    print(f"\ntrain mean UAS score: {train_uas[0]}\n\n")

    if save:
        print(f"\nSaving model: {model_path}\n")
        torch.save(model, model_path)


def run_exp():

    for i, lr in enumerate([5e-3]):
        print(f"\n\n############################### Start exp {i} with lr: {lr} ###############################\n\n")
        exp(epochs=4, lr=lr, batch_size=32, test=False, save=True, train_size="concat")
        print(f"\n\n############################### Finish exp {i} with lr: {lr} ###############################\n\n")


def write_comp(model_path, comp_path):
    comp_sentences = get_sentences(comp_path, mode="pred")
    model = torch.load(model_path, map_location=device)

    model.get_predictions(comp_sentences)

    with open('data/comp.labeled', 'w') as f:
        write_sentences(comp_sentences, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='BiLSTM Dependency Parser')
    parser.add_argument('-e', '--epochs', dest='e', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch_size', dest='bs', type=int, default=32)
    parser.add_argument('-t', '--test', dest='test', action='store_true')
    parser.add_argument('-s', '--save', dest='save', action='store_true')
    parser.add_argument('-ts', '--train_size', dest='ts', type=str, default='full')
    parser.add_argument('-me', '--multi_exp', dest='me', action='store_true')
    parser.add_argument('-wc', '--write_comp', dest='wc', action='store_true')
    args = parser.parse_args()

    if args.wc:
        write_comp(model_path='models/model.pt', comp_path='data/comp.unlabeled')
    elif args.me:
        run_exp()
    else:
        exp(args.e, args.lr, args.bs, args.test, args.save, args.ts)
