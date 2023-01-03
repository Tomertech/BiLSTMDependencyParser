from collections import Counter
import re
import pickle
from functools import wraps

# import conllu
import time
from gensim import downloader
import torchtext
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class Entry:
    def __init__(self, loc, word, pos, parent_id):
        self.loc = loc
        self.original_word = word
        self.normed_word = normalize(word)
        self.pos = pos.upper()
        self.parent_id = int(parent_id)

        self.word_vec = None
        self.pos_idx = None

        # self.cpos = "_"
        # self.lemma = "_"
        # self.feats = "_"
        # self.deps = "_"
        # self.misc = "_"
        # self.relation = "_"

        # self.pred_parent_id = None
        # self.pred_relation = None

    def __str__(self):
        values = [str(self.loc), self.normed_word, "_", "_", self.pos, "_", self.parent_id, "_", "_", "_"]
        return '\t'.join(['_' if v is None else v for v in values])


@timeit
def get_transformed_sentences(conll_path, word_vocab, pos_vocab):
    """get sentences as relevant entries"""
    words_count = Counter()
    pos_count = Counter()
    sentences = get_sentences(conll_path)

    for sentence in sentences:
        words_count.update([node.normed_word for node in sentence if isinstance(node, Entry)])
        pos_count.update([node.pos for node in sentence if isinstance(node, Entry)])

        for word in sentence:
            if "-" in word.normed_word:
                left_word = word.normed_word.split("-", 1)[0]
                right_word = word.normed_word.split("-", 1)[1]
                word.word_vec = word_vocab[left_word] + word_vocab[right_word]
            else:
                word.word_vec = word_vocab[word.normed_word]

            word.pos_idx = pos_vocab[word.pos]
            word.word_vec.to(device)

    return sentences


def get_sentences(file):
    sentences = []
    root = Entry(0, '*root*', '*root*', 0)
    curr_sentence = [root]  # add root

    with open(file, 'r') as f:
        for line in f.readlines():
            entry = line.strip().split('\t')

            if line == '\n' or line == '\t' or line == ' ' or line.startswith('#'):
                if len(curr_sentence) > 1:
                    sentences.append(curr_sentence)  # add sentence
                    curr_sentence = [root]  # make a root for the next sentence

            else:
                curr_sentence.append(Entry(loc=int(entry[0]), word=entry[1], pos=entry[3], parent_id=entry[6]))

    return sentences


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    if numberRegex.match(word):
        return "###"

    elif word.endswith(".") and len(word) > 1:
        return word[:-1].lower()

    else:
        return word.lower()


@timeit
def test_create_vocab(test_path, my_sentences):

    true_sentences = conllu.parse(open(test_path, "r").read())
    for ts, ms in zip(true_sentences, my_sentences):
        if len(ts)+1 != len(ms):
            print("Error")
            print(ts, ms)


def get_pre_trained_voacb(pre_traind_vocab_name="embeds/glove-wiki-gigaword-100"):
    # try:
    #     pre_traind_vocab = load_pickle(pre_traind_vocab_name)
    # except FileNotFoundError:
    #     pre_traind_vocab = downloader.load(pre_traind_vocab_name)
    #     save_pickle(pre_traind_vocab, pre_traind_vocab_name)
    #
    # finally:
    #     return pre_traind_vocab
    fasttext = torchtext.vocab.FastText(language='en')
    return fasttext


def get_pos_vocab():
    pos_tags = {
        "CC": 0,
        "CD": 1,
        "DT": 2,
        "EX": 3,
        "FW": 4,
        "IN": 5,
        "JJ": 6,
        "JJR": 7,
        "JJS": 8,
        "LS": 9,
        "MD": 10,
        "NN": 11,
        "NNS": 12,
        "NNP": 13,
        "NNPS": 14,
        "PDT": 15,
        "POS": 16,
        "PRP": 17,
        "PRP$": 18,
        "RB": 19,
        "RBR": 20,
        "RBS": 21,
        "RP": 22,
        "SYM": 23,
        "TO": 24,
        "UH": 25,
        "VB": 26,
        "VBD": 27,
        "VBG": 28,
        "VBN": 29,
        "VBP": 30,
        "VBZ": 31,
        "WDT": 32,
        "WP": 33,
        "WP$": 34,
        "WRB": 35,
        "(": 36,
        ")": 37,
        ",": 38,
        ".": 39,
        ":": 40,
        "``": 41,
        "''": 42,
        "$": 43,
        "#": 44,
        "*ROOT*": 45,
        "UNKNOWN": 46
    }
    return pos_tags


if __name__ == "__main__":

    pre_trained_vocab = get_pre_trained_voacb()
    pos_vocab = get_pos_vocab()
    transformed_sentences = get_transformed_sentences("data/train.labeled", word_vocab=pre_trained_vocab, pos_vocab=pos_vocab)
    test_create_vocab("data/train.labeled", transformed_sentences)

    pos_set = set()

    for sentence in transformed_sentences:
        for w in sentence:
            if w.pos not in get_pos_vocab():
                pos_set.add(w.pos)
    assert(len(pos_set) == 0)
