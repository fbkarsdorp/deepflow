import argparse
import collections
import functools
import logging
import json
import random
import shutil
from typing import Tuple, Dict, List

import gensim
import numpy as np
import sklearn.model_selection
import sklearn.metrics

import torch
from torch.nn.utils.rnn import pad_sequence

from allennlp.modules import ConditionalRandomField, TimeDistributed

import loaders
import torch_utils
import utils


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


PAD = 0
EOS = 1
BOS = 2
UNK = 3  # if it exists


def embedding_layer(
        weights: np.ndarray, n_padding_vectors: int = 4,
        padding_idx: int = None, trainable: bool = False
    ) -> torch.nn.Embedding:
    """Create an embedding layer from pre-trained gensim embeddings."""
    weights = np.vstack((np.zeros((n_padding_vectors, weights.shape[1])), weights))
    return torch.nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=not trainable)

class Tagger(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 syllable_encoder: torch.nn.Embedding,
                 embedding_dropout_p: float,
                 stress_size: int, wb_size: int, 
                 sequence_encoder) -> None:
        super(Tagger, self).__init__()

        self.embedding_dropout_p = embedding_dropout_p

        self.stress_encoder = torch.nn.Embedding(stress_size, embedding_dim, padding_idx=PAD)
        self.wb_encoder = torch.nn.Embedding(wb_size, embedding_dim, padding_idx=PAD)
        self.syllable_encoder = syllable_encoder

        self.sequence_encoder = sequence_encoder

    def pack(self, stress, wb, syllables, lengths):
        stress_embs = self.stress_encoder(stress)
        wb_embs = self.wb_encoder(wb)
        syllable_embs = self.syllable_encoder(syllables)
        embs = torch.cat([stress_embs, wb_embs, syllable_embs], 2)
        embs = torch.nn.functional.dropout(
            embs, p=self.embedding_dropout_p, training=self.training)
        embs = torch.nn.utils.rnn.pack_padded_sequence(embs, lengths, batch_first=True)
        return embs

    def unpack(self, out):
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out

    def forward(self, stress, wb, syllables, lengths, targets=None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    

class LSTMTagger(Tagger):
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 syllable_encoder: torch.nn.Embedding,                 
                 embedding_dropout_p: float,
                 stress_size: int, wb_size: int, tagset_size: int,
                 sequence_encoder) -> None:

        super(LSTMTagger, self).__init__(
            embedding_dim, hidden_dim, syllable_encoder, embedding_dropout_p,
            stress_size, wb_size, sequence_encoder)
        
        self.tag_projection_layer = torch.nn.Linear(hidden_dim, tagset_size)
        
    def forward(self, stress: torch.Tensor, wb: torch.Tensor, syllables: torch.Tensor,
                lengths: torch.Tensor, targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        encoder_out, _ = self.sequence_encoder(self.pack(stress, wb, syllables, lengths))
        tag_space = self.tag_projection_layer(self.unpack(encoder_out))
        _, preds = tag_space.max(2)
        preds = utils.chop_padding(preds.cpu().numpy(), lengths)
        output = {"tag_space": tag_space, "tags": preds}
        if targets is not None:
            loss = self.loss(tag_space, lengths, targets)
            output['loss'] = loss
        return output

    def loss(self, tag_space, lengths, targets):
        loss = torch.nn.functional.cross_entropy(
            tag_space.view(-1, tag_space.size(2)), targets.view(-1),
            size_average=False, ignore_index=0)
        loss = loss / lengths.sum().item()
        return loss


class CRFTagger(Tagger):
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 syllable_encoder: torch.nn.Embedding,
                 embedding_dropout_p: float,                 
                 stress_size: int, wb_size: int, tagset_size: int,
                 sequence_encoder: torch.nn.LSTM,
                 constraints: List[Tuple[int, int]] = None,
                 include_start_end_transitions: bool = True) -> None:
        
        super(CRFTagger, self).__init__(
            embedding_dim, hidden_dim, syllable_encoder, embedding_dropout_p,
            stress_size, wb_size, sequence_encoder)
        
        self.crf = ConditionalRandomField(
            num_tags=tagset_size, constraints=constraints,
            include_start_end_transitions=include_start_end_transitions
        )

        self.tag_projection_layer = TimeDistributed(torch.nn.Linear(hidden_dim, tagset_size))

    def forward(self, stress: torch.Tensor, wb: torch.Tensor, syllables: torch.Tensor,
                lengths: torch.Tensor, targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        encoder_out, _ = self.sequence_encoder(self.pack(stress, wb, syllables, lengths))
        tag_space = self.tag_projection_layer(self.unpack(encoder_out))
        # TODO: do we need to log_softmax?
        #tag_space = torch.nn.functional.log_softmax(tag_space, dim=2)
        mask = torch_utils.make_length_mask(
            lengths, maxlen=tag_space.size(1), device=tag_space.device)
        preds = [tag for tag, _ in self.crf.viterbi_tags(tag_space, mask)] if not self.training else []
        output = {"tag_space": tag_space, "mask": mask, "tags": preds}
        if targets is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(tag_space, targets, mask)
            output["loss"] = -log_likelihood
        return output


class Trainer:
    def __init__(self, model: LSTMTagger, train_data: loaders.DataSet,
                 dev_data: loaders.DataSet = None, test_data: loaders.DataSet = None,
                 optimizer=None, device=None, decoder=None, lr_patience: int = 5):
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
            self.optimizer, 'min', verbose=True, patience=lr_patience)

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

    def get_batch(self, batch: Dict[str, List[torch.LongTensor]]) -> Tuple[torch.Tensor]:
        lengths, perm_index = torch.LongTensor(batch['length']).sort(0, descending=True)
        stress = utils.perm_sort(batch['stress'], perm_index)
        wb = utils.perm_sort(batch['wb'], perm_index)
        syllables = utils.perm_sort(batch['syllables'], perm_index)
        targets = utils.perm_sort(batch['targets'], perm_index)
        
        stress = pad_sequence(stress, batch_first=True).to(self.device)
        wb = pad_sequence(wb, batch_first=True).to(self.device)
        syllables = pad_sequence(syllables, batch_first=True).to(self.device)
        targets = pad_sequence(targets, batch_first=True).to(self.device)

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
        y_true, y_pred, accuracy, seq_accuracy = [], [], 0, []
        run_loss, example_print = 0, 0
        n_batches = 0
        for i, batch in enumerate(data.batches()):
            stress, wb, syllables, targets, lengths = self.get_batch(batch)
            output = self.model(stress, wb, syllables, lengths, targets)
            if not test:
                run_loss += output['loss'].item()
            pred = output['tags']
            true = utils.chop_padding(targets.cpu().numpy(), lengths)
            y_true.append(true)
            y_pred.append(pred)
            accuracy += utils.accuracy_score(pred, true)
            seq_accuracy += utils.seq_accuracy_score(pred, true)
            n_batches += 1
            for i in random.sample(range(len(true)), 2):
                self.decoder.decode(syllables[i].cpu().numpy(), true[i], pred[i])
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
        logging.info('Accuracy score:          {:.3f}'.format(accuracy / n_batches))
        logging.info('Sequence Accuracy score: {:.3f}'.format(sum(seq_accuracy) / len(seq_accuracy)))

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
        with torch.no_grad():
            self.model.eval()
            self._validate(data, test=True)
        self.model.train()


