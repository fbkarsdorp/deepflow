import argparse
import collections
import functools
import logging
import json
import shutil
from typing import Tuple, Dict, List
import random

import gensim
import numpy as np
import sklearn.model_selection
import sklearn.metrics

import torch
import loaders
from allennlp.modules import ConditionalRandomField

import torch_utils


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


PAD = 0
EOS = 1
BOS = 2
UNK = 3  # if it exists


def accuracy_score(x, y):
    return sum((a == b).all() for a, b in zip(x, y)) / len(x)


def embedding_layer(
        weights: np.ndarray, n_padding_vectors: int = 4,
        padding_idx: int = None, trainable: bool = False
    ) -> torch.nn.Embedding:
    """Create an embedding layer from pre-trained gensim embeddings."""
    weights = np.vstack((np.zeros((n_padding_vectors, weights.shape[1])), weights))
    embedding_weights = torch.tensor(weights, dtype=torch.float32)
    embedding = torch.nn.Embedding(*embedding_weights.shape, padding_idx=padding_idx)
    embedding.weight = torch.nn.Parameter(embedding_weights, requires_grad=trainable)
    return embedding


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 num_layers: int, embedding_dropout_p: float,
                 syllable_encoder: torch.nn.Embedding,
                 stress_size: int, wb_size: int, tagset_size: int,
                 bidirectional: bool = True) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tagset_size = tagset_size
        self.embedding_dropout_p = embedding_dropout_p
        super(LSTM, self).__init__()

        # loss weight to remove padding
        weight = torch.ones(tagset_size)
        weight[PAD] = 0.
        self.register_buffer('weight', weight)
        
        self.stress_encoder = torch.nn.Embedding(stress_size, embedding_dim, padding_idx=PAD)
        self.wb_encoder = torch.nn.Embedding(wb_size, embedding_dim, padding_idx=PAD)
        self.syllable_encoder = syllable_encoder

        num_dirs = 1 + int(bidirectional)
        self.lstm = torch.nn.LSTM(
            embedding_dim * 2 + syllable_encoder.weight.shape[1],
            hidden_dim // num_dirs, num_layers=num_layers, batch_first=True,
            bidirectional=True)

        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, stress_sequence, wb_sequence, syllable_sequence, lengths):
        stress_embs = self.stress_encoder(stress_sequence)
        wb_embs = self.wb_encoder(wb_sequence)
        syllable_embs = self.syllable_encoder(syllable_sequence)
        embs = torch.cat([stress_embs, wb_embs, syllable_embs], 2)
        embs = torch.nn.functional.dropout(
            embs, p=self.embedding_dropout_p, training=self.training)
        lstm_out, _ = self.lstm(embs)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

    def loss(self, stress_sequence, wb_sequence, syllable_sequence, lengths, targets):
        tag_space = self(stress_sequence, wb_sequence, syllable_sequence, lengths)
        loss = torch.nn.functional.cross_entropy(
            tag_space.view(-1, tag_space.size(2)), targets.view(-1),
            weight=self.weight, size_average=False)
        loss = loss / lengths.sum().item()
        return loss
    

class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 num_layers: int, embedding_dropout_p: float,
                 syllable_encoder: torch.nn.Embedding,
                 stress_size: int, wb_size: int, tagset_size: int,
                 bidirectional: bool = True) -> None:
        
        super(LSTMTagger, self).__init__()
        self.encoder = LSTM(
            embedding_dim, hidden_dim, num_layers, embedding_dropout_p,
            syllable_encoder, stress_size, wb_size, tagset_size,
            bidirectional=True
        )

    def forward(self, stress, wb, syllables, lengths, targets=None):
        tag_space = self.encoder(stress, wb, syllables, lengths)
        _, preds = tag_space.max(2)
        preds = chop_padding(preds.cpu().numpy(), lengths)
        output = {"tag_space": tag_space, "tags": preds}
        if targets is not None:
            loss = self.encoder.loss(stress, wb, syllables, lengths, targets)
            output['loss'] = loss
        return output


class CRFTagger(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 num_layers: int, embedding_dropout_p: float,
                 syllable_encoder: torch.nn.Embedding,
                 stress_size: int, wb_size: int, tagset_size: int,
                 bidirectional: bool = True,
                 constraints: List[Tuple[int, int]] = None,
                 include_start_end_transitions: bool = True) -> None:
        
        super(CRFTagger, self).__init__()
        self.crf = ConditionalRandomField(
            num_tags=tagset_size, constraints=constraints,
            include_start_end_transitions=include_start_end_transitions
        )
        self.encoder = LSTM(
            embedding_dim, hidden_dim, num_layers, embedding_dropout_p,
            syllable_encoder, stress_size, wb_size, tagset_size,
            bidirectional=True
        )

    def forward(self, stress, wb, syllables, lengths, targets=None):
        tag_space = self.encoder(stress, wb, syllables, lengths)
        # tag_space = torch.nn.functional.log_softmax(tag_space, dim=2)
        mask = torch_utils.make_length_mask(
            lengths, maxlen=tag_space.size(1), device=tag_space.device).float()     
        preds = self.crf.viterbi_tags(tag_space, mask)
        output = {"tag_space": tag_space, "mask": mask, "tags": preds}
        if targets is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(tag_space, targets, mask)
            output["loss"] = -log_likelihood
        return output


def chop_padding(samples, lengths):
    return [samples[i, 0:lengths[i]] for i in range(samples.shape[0])]
    

class Trainer:
    def __init__(self, model: LSTMTagger, train_data: loaders.DataSet,
                 dev_data: loaders.DataSet = None, test_data: loaders.DataSet = None,
                 optimizer=None, device=None, decoder=None):
        self.logger = logging.getLogger('Trainer({})'.format(model.__class__.__name__))
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.decoder = decoder
        self.best_loss = np.inf
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', verbose=True, patience=5)

    def train(self, epochs: int = 10) -> None:
        self.logger.info("Working on {}".format(self.device))
        for epoch in range(1, epochs + 1):
            self.logger.info('Starting epoch {} with lr={}'.format(
                epoch, self.optimizer.param_groups[0]['lr']))
            self._train_batches(epoch, epochs)
            self.logger.info('Finished epoch {}'.format(epoch))
            if self.dev_data is not None:
                with torch.no_grad():
                    self.model.eval() # set all trainable attributes to False
                    self._validate(self.dev_data)
                    self.model.train() # set all trainable attributes back to True

    def get_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        stress = batch['stress'].to(self.device)
        wb = batch['wb'].to(self.device)
        syllables = batch['syllables'].to(self.device)
        targets = batch['beatstress'].to(self.device)
        lengths, perm_index = batch['length'].sort(0, descending=True)

        stress = stress[perm_index]
        wb = wb[perm_index]
        syllables = syllables[perm_index]
        targets = targets[perm_index]

        return stress, wb, syllables, targets, lengths

    def _train_batches(self, epoch: int, epochs: int) -> None:
        epoch_loss = 0
        n_batches = 0
        for i, batch in enumerate(self.train_data.batches()):
            stress, wb, syllables, targets, lengths = self.get_batch(batch)
            
            self.model.zero_grad()

            output = self.model(stress, wb, syllables, lengths, targets)
            output['loss'].backward()
            self.optimizer.step()
            loss = output['loss'].item()
            epoch_loss += loss

            n_batches += 1
            
            if i > 0 and i % 50 == 0:
                logging.info(
                    'Epoch [{}/{}], Step [{}]'.format(
                        epoch, epochs, i))
                logging.info(
                    'Loss: {}'.format(epoch_loss / i))
    
    def _validate(self, data: loaders.DataSet, test: bool = False) -> None:
        self.logger.info('Validating model')
        y_true, y_pred, accuracy, baccuracy = [], [], 0, 0
        run_loss, example_print = 0, 0
        n_batches = 0
        for i, batch in enumerate(data.batches()):
            stress, wb, syllables, targets, lengths = self.get_batch(batch)
            output = self.model(stress, wb, syllables, lengths, targets)
            if not test:
                run_loss += output['loss'].item()
            batch_size = stress.size(0)
            pred = output['tags']
            true = chop_padding(targets.cpu().numpy(), lengths)
            y_true.append(true)
            y_pred.append(pred)
            accuracy += accuracy_score(pred, true)
            n_batches += 1
        if not test:
            logging.info('Validation Loss: {}'.format(run_loss / n_batches))
            closs = run_loss / n_batches
            self.scheduler.step(closs)
            self.save_checkpoint(closs < self.best_loss)
            if closs < self.best_loss:
                self.best_loss = closs
        p, r, f, s = sklearn.metrics.precision_recall_fscore_support(
            np.hstack(np.hstack(y_true)), np.hstack(np.hstack(y_pred)))
        for i in range(p.shape[0]):
            logging.info('Validation Scores: c={} p={:.3f}, r={:.3f}, f={:.3f}, s={}'.format(
                i, p[i], r[i], f[i], s[i]))
        logging.info('Accuracy score: {:.3f}'.format(accuracy / n_batches))

    def save_checkpoint(self, is_best: bool, filename: str = 'checkpoint.pth.tar') -> None:
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def test(self, data: loaders.DataSet = None, statefile: str = None):
        if data is None and self.test_data is not None:
            data = self.test_data
        if statefile is not None:
            checkpoint = torch.load(statefile)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._validate(data, test=True)
            
