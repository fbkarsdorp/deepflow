import argparse
import collections
import functools
import logging
import json
import shutil
import random

import gensim
import numpy as np
import sklearn.model_selection
import sklearn.metrics

import torch
import loaders

import torch_utils

random.seed(1983)
np.random.seed(1983)
torch.manual_seed(1983)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


PAD = 0
EOS = 1
BOS = 2
UNK = 3  # if it exists


def embedding_layer(weights, n_padding_vectors=4, trainable=False):
    """Create an embedding layer from pre-trained gensim embeddings."""
    weights = np.vstack((np.zeros((n_padding_vectors, weights.shape[1])), weights))
    embedding_weights = torch.tensor(weights, dtype=torch.float32)
    embedding = torch.nn.Embedding(*embedding_weights.shape, padding_idx=PAD)
    embedding.weight = torch.nn.Parameter(embedding_weights, requires_grad=trainable)
    return embedding


class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding_dropout_p,
                 stress_size, wb_size, syllable_encoder, tagset_size, batch_size,
                 bidirectional=True):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.tagset_size = tagset_size
        self.embedding_dropout_p = embedding_dropout_p
        super(LSTMTagger, self).__init__()

        # loss weight to remove padding
        weight = torch.ones(tagset_size)
        weight[PAD] = 0.
        self.register_buffer('weight', weight)
        self.stress_encoder = torch.nn.Embedding(
            stress_size, embedding_dim, padding_idx=PAD)
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
        # packed = torch.nn.utils.rnn.pack_padded_sequence(
        #     embs, lengths, batch_first=True)
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

    def predict(self, stress_sequence, wb_sequence, syllable_sequence, lengths):
        tag_space = self(stress_sequence, wb_sequence, syllable_sequence, lengths)
        # tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        pred = tag_space.view(-1, tag_space.size(2)).data.cpu().numpy().argmax(1)

        return pred


class CRFLSTMTagger(LSTMTagger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # matrix of transition scores from j to i
        trans = torch.randn(self.tagset_size, self.tagset_size)
        trans[BOS, :] = -10000. # no transition to <bos>
        trans[:, EOS] = -10000. # no transition from <eos> except to <pad>
        trans[:, PAD] = -10000. # no transition from <pad> except to <pad>
        trans[PAD, :] = -10000.
        trans[PAD, EOS] = 0.
        trans[PAD, PAD] = 0.
        self.trans = torch.nn.Parameter(trans)

    def forward(self, stress_sequence, wb_sequence, syllable_sequence, lengths):
        tag_space = super().forward(
            stress_sequence, wb_sequence, syllable_sequence, lengths)
        tag_space = torch.nn.functional.log_softmax(tag_space, dim=2)
        return tag_space

    def loss(self, stress_sequence, wb_sequence, syllable_sequence, lengths, targets):
        tag_space = self(stress_sequence, wb_sequence, syllable_sequence, lengths)

        # mask on padding
        mask = torch_utils.make_length_mask(
            lengths, maxlen=tag_space.size(1), device=tag_space.device).float()
        tag_space = tag_space * mask.unsqueeze(2).expand_as(tag_space)

        Z = self.partition(tag_space, mask)
        score = self.score(tag_space, mask, targets)

        return torch.mean(Z - score)

    def partition(self, tag_space, mask):
        tag_space = tag_space[:, 1:]
        batch, seq_len, vocab = tag_space.size()

        # initialize forward variables in log space
        Z = tag_space.new(batch, vocab).fill_(-10000.)
        Z[:, BOS] = 0.

        # iterate through the sequence
        for t in range(seq_len): 
            Z_t = Z.unsqueeze(1).expand(-1, vocab, vocab)
            emit = tag_space[:, t].unsqueeze(-1).expand_as(Z_t)
            trans = self.trans.unsqueeze(0).expand_as(Z_t)
            Z_t = torch_utils.log_sum_exp(Z_t + emit + trans)
            mask_t = mask[:, t].unsqueeze(-1).expand_as(Z)
            Z = Z_t * mask_t + Z * (1 - mask_t)

        Z = torch_utils.log_sum_exp(Z)

        return Z

    def score(self, tag_space, mask, targets):
        # calculate the score of a given sequence
        batch, seq_len, vocab = tag_space.size()
        score = tag_space.new(batch).fill_(0.)

        # targets = torch_utils.prepad(targets, pad=BOS)
        targets = torch.nn.functional.pad(targets, (1, 0), value=BOS)

        # iterate over sequence
        for t in range(seq_len):
            emit = tag_space[:, t, targets[:, t+1]].diag()
            trans = self.trans[targets[:, t+1], targets[:, t]]
            score = score + emit + (trans * mask[:, t])

        return score

    def predict(self, stress_sequence, wb_sequence, syllable_sequence, lengths):
        tag_space = self(stress_sequence, wb_sequence, syllable_sequence, lengths)

        maxlen = tag_space.size(1)
        vocab = tag_space.size(-1)

        hyps, scores = [], []

        # TODO: don't iterate over batches
        # iterate over batches
        for item, length in zip(tag_space, lengths.tolist()):
            # (seq_len x batch x vocab) => (real_len x vocab)
            item = item[:length]
            bptr = []
            score = tag_space.new(vocab).fill_(-10000.)
            score[BOS] = 0.

            # iterate over sequence
            for emit in item:
                bptr_t, score_t = [], []

                # TODO: don't iterate over tags
                # for each next tag
                for i in range(vocab):
                    prob, best = torch.max(score + self.trans[i], dim=0)
                    bptr_t.append(best.item())
                    score_t.append(prob)
                # accumulate
                bptr.append(bptr_t)
                score = torch.stack(score_t) + emit

            score, best = torch.max(score, dim=0)
            score, best = score.item(), best.item()

            # back-tracking
            hyp = [best]
            for bptr_t in reversed(bptr):
                hyp.append(bptr_t[best])
            hyp = list(reversed(hyp[:-1]))
            hyp = hyp + [PAD] * (maxlen - len(hyp))

            hyps.append(np.array(hyp))

        return np.array(hyps)


def chop_padding(samples, lengths):
    return [samples[i, 0:lengths[i] - 1] for i in range(samples.shape[0])]
    

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

    def train(self, epochs=10):
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

    def _train_batches(self, epoch, epochs):
        epoch_loss = 0
        n_samples = 0
        for i, batch in enumerate(self.train_data.batches()):
            stress = batch['stress'].to(self.device)
            wb = batch['wb'].to(self.device)
            syllables = batch['syllables'].to(self.device)
            targets = batch['beatstress'].to(self.device)
            lengths, perm_index = batch['length'].sort(0, descending=True)

            stress = stress[perm_index]
            wb = wb[perm_index]
            syllables = syllables[perm_index]
            targets = targets[perm_index]
            
            self.model.zero_grad()

            loss = self.model.loss(stress, wb, syllables, lengths, targets)
            loss.backward()
            self.optimizer.step()
            loss = loss.item()
            epoch_loss += loss

            batch_size = stress.size(0)
            n_samples += batch_size
            
            if i > 0 and i % 50 == 0:
                logging.info(
                    'Epoch [{}/{}], Step [{}]'.format(
                        epoch, epochs, i))
                logging.info(
                    'Loss: {}'.format(epoch_loss / i))
    
    def _validate(self, data: loaders.DataSet, test=False):
        self.logger.info('Validating model')
        y_true, y_pred, accuracy, baccuracy = [], [], 0, 0
        run_loss, example_print = 0, 0
        n_batches = 0
        for i, batch in enumerate(data.batches()):
            stress = batch['stress'].to(self.device)
            wb = batch['wb'].to(self.device)
            syllables = batch['syllables'].to(self.device)
            targets = batch['beatstress'].to(self.device)
            lengths, perm_index = batch['length'].sort(0, descending=True)
            stress = stress[perm_index]
            wb = wb[perm_index]
            syllables = syllables[perm_index]
            targets = targets[perm_index]

            if not test:
                loss = self.model.loss(stress, wb, syllables, lengths, targets).item()
                run_loss += loss

            # collect predictions
            batch_size = stress.size(0)
            pred = self.model.predict(stress, wb, syllables, lengths)
            true = targets.view(-1).cpu().numpy()
            true = true.reshape(batch_size, stress.size(1))
            pred = pred.reshape(batch_size, stress.size(1))
            pred = chop_padding(pred, lengths)
            true = chop_padding(true, lengths)
            y_true.append(true)
            y_pred.append(pred)
            accuracy += sum((t == p).all() for t, p in zip(pred, true)) / batch_size
            n_batches += 1
            # if i % 10 == 0:
            #     for elt in np.random.randint(0, batch_size, 2):
            #         self.decoder.decode(syllables[elt], true[elt], pred[elt])
        if not test:
            logging.info('Validation Loss: {}'.format(run_loss / n_batches))
            closs = run_loss / n_samples
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

    def save_checkpoint(self, is_best, filename='checkpoint.pth.tar'):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def test(self, data: loaders.DataSet = None, statefile=None):
        if data is None and self.test_data is not None:
            data = self.test_data
        if statefile is not None:
            checkpoint = torch.load(statefile)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._validate(data)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--train_file', required=True, type=str)
    parser.add_argument('--dev_file', required=True, type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--pretrained_embeddings', required=True, type=str)
    parser.add_argument('--retrain_embeddings', action='store_true')
    parser.add_argument('--dev_size', default=0.05, type=float)
    parser.add_argument('--test_size', default=0.05, type=float)
    args = parser.parse_args()
    
    syllable_vocab, syllable_vectors = loaders.load_gensim_embeddings(args.pretrained_embeddings)

    stress_encoder = loaders.Encoder('stress', preprocessor=loaders.normalize_stress)
    beat_encoder = loaders.Encoder('beatstress', preprocessor=loaders.normalize_stress)
    syllable_encoder = loaders.Encoder('syllables', vocab=syllable_vocab, fixed_vocab=True)
    wb_encoder = loaders.Encoder('syllables', preprocessor=loaders.word_boundaries)

    syllable_embeddings = embedding_layer(
        syllable_vectors, n_padding_vectors=len(syllable_encoder.reserved_tokens),
        trainable=args.retrain_embeddings)    

    train = loaders.DataSet(args.train_file, stress=stress_encoder, beatstress=beat_encoder,
                            wb=wb_encoder, syllables=syllable_encoder, batch_size=args.batch_size)
    dev = loaders.DataSet(args.dev_file, stress=stress_encoder, beatstress=beat_encoder,
                          wb=wb_encoder, syllables=syllable_encoder, batch_size=args.batch_size)
    if args.test_file is not None:
        test = loaders.DataSet(args.test_file, stress=stress_encoder, beatstress=beat_encoder,
                               wb=wb_encoder, syllables=syllable_encoder, batch_size=args.batch_size)    
    # construct a layer holding the pre-trained word2vec embeddings
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the tagger
    tagger = CRFLSTMTagger(
        args.emb_dim, args.hid_dim, args.num_layers, args.dropout, stress_encoder.size() + 2,
        wb_encoder.size() + 2, syllable_embeddings, beat_encoder.size() + 2, args.batch_size, bidirectional=True)
    print(tagger)
    # The Adam optimizer seems to be working fine. Make sure to exlcude the pretrained
    # embeddings from optimizing
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, tagger.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay
    ) # RMSprop
    tagger.to(device)
    trainer = Trainer(
        tagger, train, dev, test, optimizer, device=device)
    trainer.train(epochs=args.epochs)
    trainer.test(statefile='model_best.pth.tar')
