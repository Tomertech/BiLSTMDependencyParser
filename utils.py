from typing import List
import re
import pickle
from functools import wraps
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
# import conllu
import time
from gensim import downloader
import torchtext
import torch

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

glove = torchtext.vocab.GloVe(name='6B', dim=100)

pos_list = [
        "CC",
        "CD",
        "DT",
        "EX",
        "FW",
        "IN",
        "JJ",
        "JJR",
        "JJS",
        "LS",
        "MD",
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "PDT",
        "POS",
        "PRP",
        "PRP$",
        "RB",
        "RBR",
        "RBS",
        "RP",
        "SYM",
        "TO",
        "UH",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "WDT",
        "WP",
        "WP$",
        "WRB",
        "(",
        ")",
        ",",
        ".",
        ":",
        "``",
        "''",
        "$",
        "#",
        "*ROOT*"
    ]
pos_to_idx = defaultdict(lambda: len(pos_list))
for pos in pos_list:
    pos_to_idx[pos] = len(pos_to_idx)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ General ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Task related ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_arcs(n_words):
    """ Get all possible arcs of the sentence, (v_i, v_j) where i != j and v_j is the root """
    arcs = list(combinations(list(range(0, n_words)), 2))  # all possible arcs -> between words in sentence and root
    arcs += list(combinations(reversed(list(range(1, n_words))), 2))  # all possible arcs -> between words in sentence
    return arcs


class Sentence:
    def __init__(self, words: list, poss: list, parent_ids: list):
        self.original_words = words
        self.normalized_words = [word.lower() for word in words]
        self.poss = [pos.upper() for pos in poss]
        self.parent_ids = torch.tensor(parent_ids, device=device, requires_grad=False)
        self.words_embeddings = get_embeddings(words)
        self.poss_indices = torch.tensor([pos_to_idx[pos] for pos in poss], device=device, requires_grad=False)
        self.preds_parents_ids = None
        # get all possible arcs of the sentence, (v_i, v_j) where i != j and v_j != 0
        # because root can be a parent of any word, buy not vice versa
        self.all_possible_arcs = get_arcs(len(words))

    def __len__(self):
        return len(self.original_words)

    def __str__(self):
        return ' '.join(self.original_words)


def get_embeddings(words: list):
    words_embeddings = []

    for word in words:
        if word in glove.stoi.keys():
            words_embeddings.append(glove[normalize(word)])
        elif word == "--" or word == "-":
            words_embeddings.append(glove[normalize("-")])
        elif "-" in word:
            seperated_words = list(set(word.split("-")))  # remove duplicates
            seperated_words = [normalize(word) for word in seperated_words]
            words_embeddings.append(glove.get_vecs_by_tokens(seperated_words).sum(dim=0))
        else:
            words_embeddings.append(glove[normalize(word)])  # if word not in glove, return zeroes vector

    return torch.stack(words_embeddings).to(device)


@timeit
def get_sentences(file, mode='train'):
    sentences = []
    words, poss, parent_ids = ['*root*'], ['*root*'], [0]

    with open(file, 'r') as f:
        for line in f.readlines():
            if line == '\n' or line == '\t' or line == ' ' or line.startswith('#'):
                sentences.append(Sentence(words, poss, parent_ids))
                words, poss, parent_ids = ['*root*'], ['*root*'], [0]
            else:
                entry = line.strip().split('\t')
                words.append(entry[1])
                poss.append(entry[3])
                if mode == 'train':
                    parent_ids.append(int(entry[6]))

    return sentences


def write_sentences(sentences: List[Sentence], f):
    with open(f, 'w') as f:
        for sentence in sentences:
            words = sentence.original_words[1:]
            poss = sentence.poss[1:]
            parent_ids = sentence.preds_parents_ids[1:]

            for i, (word, pos, parent_id) in enumerate(zip(words, poss, parent_ids)):
                f.write(f"{i+1}\t{word}\t_\t{pos}\t_\t_\t{parent_id}\t_\t_\t_")
                f.write('\n')
            f.write('\n')


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    if numberRegex.match(word):
        return "8"  # just a number that appears in GloVe

    elif word.endswith(".") and len(word) > 1:
        return word[:-1].lower()

    else:
        return word.lower()


def plot_stats(loss_list, uas_list, title):
    fig = plt.figure(figsize=(17, 5))
    fig.suptitle(f'{title} Curves', fontsize=16)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(np.arange(len(loss_list)), loss_list)
    ax1.set_title('Cross Entropy Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(np.arange(len(uas_list)), uas_list)
    ax2.set_title('UAS')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs')

    plt.savefig(f'{title}_curves.jpg')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ test functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_create_vocab(test_path, my_sentences):

    true_sentences = conllu.parse(open(test_path, "r").read())
    for ts, ms in zip(true_sentences, my_sentences):
        if len(ts)+1 != len(ms):
            print("Error")
            print(ts, ms)


def test_get_embeddings():
    words = ["Hello", "Hello-world", "hello-World-again-again", "-", "--"]
    words_embeddings = get_embeddings(words)
    print(words_embeddings)


def test_get_sentences():
    sentences = get_sentences("data/train.labeled")
    print(sentences[0])


def test_write_sentences():
    sentences = get_sentences("data/train_short_5.labeled")
    write_sentences(sentences, "data/train_short_5_write_sentences")


if __name__ == "__main__":
    test_write_sentences()
    # arcs = get_arcs(5)
    # test_get_sentences()
    # test_get_embeddings()


