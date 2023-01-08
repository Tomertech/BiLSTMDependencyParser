import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import torch.nn as nn
import torch
from typing import List
from tqdm import tqdm
from chu_liu_edmonds import decode_mst
from utils import pos_to_idx, Sentence, plot_stats
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class EncoderBiLSTM(nn.Module):
    # default values taken from the paper:
    # "Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations" page 324
    def __init__(self, word_embedding_dim, pos_vocab_size, pos_embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim, device=device)
        self.lstm = nn.LSTM(input_size=word_embedding_dim+pos_embedding_dim, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=True, device=device)

    def forward(self, sentence: Sentence):

        pos_embedding = self.pos_embedding(sentence.poss_indices)

        # concatenate word and pos embeddings
        sequence = torch.cat((sentence.words_embeddings, pos_embedding), 1)
        v, _ = self.lstm(sequence)  # BiLSTM
        return v


class MSTDependencyParser(nn.Module):
    def __init__(self, sentences: List[Sentence], pos_emb_dim=25, hidden_dim_fc=100, hidden_dim_lstm=125, num_layers=2):
        super().__init__()
        self.sentences = sentences
        self.word_embedding_dim = sentences[0].words_embeddings.shape[1]
        self.pos_emb_dim = pos_emb_dim
        self.encoder = EncoderBiLSTM(word_embedding_dim=self.word_embedding_dim, pos_vocab_size=len(pos_to_idx),
                                     pos_embedding_dim=self.pos_emb_dim, hidden_size=hidden_dim_lstm,
                                     num_layers=num_layers)

        self.loss = nn.CrossEntropyLoss()

        self.fc_1 = nn.Linear(hidden_dim_lstm * 2 * 2, hidden_dim_fc, device=device)  # first *2 for concat, second *2 for bidirectional
        self.fc_2 = nn.Linear(hidden_dim_fc, 1, device=device)
        self.activation = nn.Tanh()

    def forward(self, sentence: Sentence):

        words_as_features = self.encoder(sentence)
        sen_len = len(sentence)
        heads = words_as_features.repeat([sen_len, 1]).reshape(sen_len, sen_len, -1)
        modifiers = heads.clone().transpose(0, 1)
        x = torch.cat((heads, modifiers), 2)
        scores_mat = self.fc_2(self.activation(self.fc_1(x))).squeeze()

        # Calculate the negative log likelihood loss described above
        loss = self.loss(scores_mat, sentence.parent_ids)

        return loss, scores_mat

    def train_model(self, epochs, lr, batch_size, test_sentences):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        train_loss, train_uas, test_loss, test_uas = [], [], [], []

        for epoch in range(epochs):
            start_time = time.perf_counter()
            for batch_idx, sentence in enumerate(tqdm(self.sentences, position=0, desc="sentence", leave=False)):
                loss, _ = self.forward(sentence)
                loss.backward()

                if (batch_idx + 1) % batch_size == 0 or (batch_idx + 1) == len(self.sentences):
                    optimizer.step()
                    optimizer.zero_grad()

            total_time = time.perf_counter() - start_time
            epoc_train_uas, epoc_train_loss = self.get_predictions(self.sentences)
            epoc_test_uas, epoc_test_loss = self.get_predictions(test_sentences)
            train_loss.append(epoc_train_loss)
            train_uas.append(epoc_train_uas)
            test_loss.append(epoc_test_loss)
            test_uas.append(epoc_test_uas)

            print(f'\nEpoch {epoch} loss train: {round(epoc_train_loss, 4)}, UAS train: {round(epoc_train_uas, 4)} '
                  f'loss test: {round(epoc_test_loss, 4)}, UAS test: {round(epoc_test_uas, 4)} '
                  f'Took {total_time:.3f} seconds')

            plot_stats(train_loss, train_uas, f"train - epochs: {epochs} lr: {lr}, batch_size: {batch_size}")
            plot_stats(test_loss, test_uas, f"test - epochs: {epochs} lr: {lr}, batch_size: {batch_size}")

    def eval_model(self, sentence):
        with torch.no_grad():
            loss, score_mat = self.forward(sentence)
            score_mat = score_mat.T  # in order to get matrix of scores of edges (i, j)
            # make probability
            score_mat.fill_diagonal_(-torch.inf)
            score_mat[:, 0] = -torch.inf
            score_mat = torch.softmax(score_mat, dim=0)  # dim=0 because we want to softmax each row
            predicted_tree = decode_mst(score_mat.cpu().detach().numpy(), length=len(score_mat), has_labels=False)[0]
        return predicted_tree, loss.item()

    def get_predictions(self, sentences: List[Sentence]):
        """"inserts to each sentence the predicted tree and returns the UAS score"""
        self.eval()
        self.encoder.eval()

        sum_pred_correct = 0
        sum_sentences_lens = 0
        sum_loss = 0.0

        for sentence in tqdm(sentences):
            true_labels = sentence.parent_ids
            pred_labels, sen_loss = self.eval_model(sentence)
            sum_loss += sen_loss
            correct = sum([1 if true_label == pred_label else 0 for true_label, pred_label in
                           zip(true_labels[1:], pred_labels[1:])])
            sentence.preds_parents_ids = pred_labels
            sum_pred_correct += correct
            sum_sentences_lens += len(true_labels[1:])

        uas = sum_pred_correct / sum_sentences_lens
        loss = sum_loss / len(sentences)

        self.train()
        self.encoder.train()

        return uas, loss
