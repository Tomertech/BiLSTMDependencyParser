import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import torch.nn as nn
import torch
from itertools import combinations
from tqdm import tqdm
from chu_liu_edmonds import decode_mst


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class EncoderBiLSTM(nn.Module):
    # default values taken from the paper:
    # "Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations" page 324
    def __init__(self, word_vocab, word_embedding_dim, pos_vocab, pos_embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.words_embeddings = word_vocab
        self.pos_vocab = pos_vocab
        self.pos_embedding = nn.Embedding(len(pos_vocab), pos_embedding_dim, device=device)
        self.lstm = nn.LSTM(input_size=word_embedding_dim+pos_embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=True, device=device)

    def forward(self, sentence: list):
        words_embedding = []
        pos_embeddings = []

        for word in sentence:
            words_embedding.append(word.word_vec.to(device))
            pos_embeddings.append(self.pos_embedding(torch.tensor(word.pos_idx, device=device)))

        tensor_words_embedding = torch.stack(words_embedding)
        tensor_pos_embedding = torch.stack(pos_embeddings)

        # concatenate word and pos embeddings
        sequence = torch.cat((tensor_words_embedding, tensor_pos_embedding), 1).to(device)
        v, _ = self.lstm(sequence)  # BiLSTM
        return v


class MSTDependencyParser(nn.Module):
    def __init__(self, sentences, word_vocab, pos_vocab, pos_emb_dim=25, hidden_dim_fc=100, hidden_dim_lstm=125, num_layers=2):
        super().__init__()
        self.sentences = sentences
        self.word_vocab = word_vocab  # embedding Fasttext / Glove
        self.pos_vocab = pos_vocab
        self.word_embedding_dim = self.word_vocab.dim
        self.pos_emb_dim = pos_emb_dim
        self.encoder = EncoderBiLSTM(word_vocab=self.word_vocab, word_embedding_dim=self.word_embedding_dim,
                                     pos_vocab=self.pos_vocab, pos_embedding_dim=self.pos_emb_dim,
                                     hidden_size=hidden_dim_lstm, num_layers=num_layers)

        self.loss = nn.CrossEntropyLoss()

        self.fc_1 = nn.Linear(hidden_dim_lstm * 2 * 2, hidden_dim_fc, device=device)  # first *2 for concat, second *2 for bidirectional
        self.fc_2 = nn.Linear(hidden_dim_fc, 1, device=device)
        self.activation = nn.Tanh()

    @staticmethod
    def _get_arcs(n_words):
        """ Get all possible arcs of the sentence, (v_i, v_j) where i != j and v_j is the root """
        arcs = list(combinations(list(range(0, n_words)), 2))  # all possible arcs -> between words in sentence and root
        arcs += list(combinations(reversed(list(range(1, n_words))), 2))  # all possible arcs -> between words in sentence
        return arcs

    @staticmethod
    def _get_parents_ids(sentence):
        parents_ids = []  # list of true heads for each word in sentence
        for word in sentence:
            parents_ids.append(word.parent_id)
        return parents_ids

    def forward(self, sentence: list):

        # words_indices, pos_indices, parents_ids = self._get_sentence_attributes(sentence)
        words_as_features = self.encoder(sentence)  # (n_words, 2*hidden_dim_lstm)

        # get all possible arcs of the sentence, (v_i, v_j) where i != j and v_j != 0
        # because root can be a parent of any word, buy not vice versa
        arcs = self._get_arcs(len(sentence))

        # Get score for each possible edge in the parsing graph by construct score matrix
        scores_mat = torch.zeros((len(sentence), len(sentence)), device=device)

        # Fill matrix with scores
        for i, j in arcs:
            score = self.fc_2(self.activation(self.fc_1(torch.cat((words_as_features[i], words_as_features[j]), 0)))).to(device)
            scores_mat[i][j] = score

        # transpose scores matrix to get the correct shape,  we want the score of each word's parent
        words_parents_scores = scores_mat.T

        # Calculate the negative log likelihood loss described above
        parents_ids = self._get_parents_ids(sentence)
        loss = self.loss(words_parents_scores, torch.tensor(parents_ids, device=device))

        return loss, scores_mat

    def train_model(self, epochs, lr, batch_size):
        running_loss = 0.0
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            start_time = time.perf_counter()
            for batch_idx, sentence in enumerate(tqdm(self.sentences, position=0, desc="sentence", leave=False)):
                loss, _ = self.forward(sentence)
                loss.backward()
                running_loss += loss.item()

                if (batch_idx + 1) % batch_size == 0 or (batch_idx + 1) == len(self.sentences):
                    optimizer.step()
                    optimizer.zero_grad()

            total_time = time.perf_counter() - start_time
            print(f'\nEpoch {round(epoch, 4)} loss: {running_loss/len(self.sentences)}, Took {total_time:.3f} seconds')
            running_loss = 0.0

    def eval_model(self, sentence):

        _, score_mat = self.forward(sentence)
        # make probability
        score_mat.fill_diagonal_(-torch.inf)
        score_mat[:, 0] = -torch.inf
        score_mat = torch.softmax(score_mat, dim=1)
        predicted_tree = decode_mst(score_mat.cpu().detach().numpy(), length=len(score_mat), has_labels=False)[0]
        return predicted_tree

    def get_uas_corpus(self, sentences):
        """this is how they asked to calculate the UAS score"""
        self.eval()
        self.encoder.eval()
        sum_pred_correct = 0
        sum_sentences_lens = 0

        for sentence in tqdm(sentences):
            true_labels = []

            for word in sentence:
                true_labels.append(word.parent_id)

            pred_labels = self.eval_model(sentence)
            correct = sum([1 if true_label == pred_label else 0 for true_label, pred_label in
                           zip(true_labels[1:], pred_labels[1:])])

            sum_pred_correct += correct
            sum_sentences_lens += len(true_labels[1:])

        return sum_pred_correct / sum_sentences_lens
