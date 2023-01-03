import unittest
import torch
from model import MSTDependencyParser, EncoderBiLSTM


class TestModel(unittest.TestCase):

    def test_get_arcs(self):
        arcs = MSTDependencyParser._get_arcs(n_words=3)
        self.assertEqual(arcs, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 2), (3, 1), (2, 1)])

    # def test_encoder(self):
    #     encoder = EncoderBiLSTM(word_vocab_size=100, word_embedding_dim=100, pos_vocab_size=6, pos_embedding_dim=25,
    #                             hidden_dim=100, num_layers=2)
    #     word_idx = torch.LongTensor([1, 2, 3, 4, 5])
    #     pos_idx = torch.LongTensor([1, 2, 3, 4, 5])
    #     v = encoder(word_idx, pos_idx)
    #     self.assertEqual(v.shape, (5, 200))
    #
    # def test_model(self):
    #     model = MSTDependencyParser(words_count={'a': 10, 'b': 5}, voc=['a', 'b'], pos=['NNP'], word_emb_dim=100, pos_emb_dim=100, hidden_dim=100,
    #                                 num_layers=2)
    #     sentence = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
    #     loss = model(sentence)
    #     self.assertEqual(loss.shape, ())
