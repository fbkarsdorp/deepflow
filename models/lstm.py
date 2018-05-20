import argparse
import collections
import logging
import shutil
import random

import gensim
import numpy as np
import sklearn.model_selection
import sklearn.metrics

import torch
from torch.utils.data import DataLoader

random.seed(1685)
np.random.seed(1685)
torch.manual_seed(1685)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def embedding_layer(weights, n_padding_vectors=4, trainable=False):
    """Create an embedding layer from pre-trained gensim embeddings."""
    weights = np.vstack((np.zeros((n_padding_vectors, weights.shape[1])), weights))
    embedding_weights = torch.FloatTensor(weights)
    embedding = torch.nn.Embedding(*embedding_weights.shape, padding_idx=0)
    embedding.weight = torch.nn.Parameter(embedding_weights, requires_grad=trainable)
    return embedding


def load_gensim_embeddings(fpath: str):
    model = gensim.models.KeyedVectors.load(fpath)
    model.init_sims(replace=True)
    return model.index2word, model.vectors


class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding_dropout_p,
                 stress_size, pos_size, syllable_encoder, tagset_size, batch_size, device):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.stress_encoder = torch.nn.Embedding(stress_size, embedding_dim, padding_idx=0).to(device)
        self.pos_encoder = torch.nn.Embedding(pos_size, embedding_dim, padding_idx=0).to(device)
        self.syllable_encoder = syllable_encoder
        self.dropout = torch.nn.Dropout(p=embedding_dropout_p)
        self.lstm = torch.nn.LSTM(
            embedding_dim * 2 + syllable_encoder.weight.shape[1],
            hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim)).to(self.device),
                torch.autograd.Variable(
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim)).to(self.device))

    def forward(self, stress_sequence, pos_sequence, syllable_sequence):
        stress_embeddings = self.stress_encoder(stress_sequence)
        # pos_embeddings = self.pos_encoder(pos_sequence)
        syllable_embeddings = self.syllable_encoder(syllable_sequence)
        embeddings = self.dropout(
            torch.cat((stress_embeddings, syllable_embeddings), 2))
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores


class BiLSTMTagger(LSTMTagger):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding_dropout_p,
                 stress_size, pos_size, syllable_encoder, tagset_size, batch_size, device):
        super(BiLSTMTagger, self).__init__(
            embedding_dim, hidden_dim, num_layers,
            embedding_dropout_p, stress_size, pos_size,
            syllable_encoder, tagset_size, batch_size, device
        )

        self.lstm = torch.nn.LSTM(
            embedding_dim + syllable_encoder.weight.shape[1], hidden_dim // 2,
            num_layers=num_layers, bidirectional=True, batch_first=True
        )

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(
                    torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2)).to(self.device),
                torch.autograd.Variable(
                    torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2)).to(self.device)
        )


class VerseData(torch.utils.data.Dataset):
    def __init__(self, fpath=None, data=None, transform=lambda x: x, shuffle_data=True):
        if data is None:
            self.data_samples = [line.strip().split(';') for line in open(fpath)]
        else:
            self.data_samples = data
        if shuffle_data:
            random.shuffle(self.data_samples)
        self.transform = transform

    def __len__(self):
        return len(self.data_samples)

    def __iter__(self):
        for sample in self.data_samples:
            yield self.transform(sample)

    def __getitem__(self, idx):
        return self.transform(self.data_samples[idx])

    def train_dev_test_split(self, dev_size=0.05, test_size=0.05):
        X_train, X_dev = sklearn.model_selection.train_test_split(
            self.data_samples, test_size=dev_size)
        X_train, X_test = sklearn.model_selection.train_test_split(
            X_train, test_size=test_size)
        return (VerseData(data=X_train, transform=self.transform),
                VerseData(data=X_dev, transform=self.transform),
                VerseData(data=X_test, transform=self.transform))


def indexer(pre_fill=None):
    d = collections.defaultdict()
    d.default_factory = lambda: len(d)
    if pre_fill is not None:
        for item in pre_fill:
            d[item]
    return d


class Sample2Tensor:
    def __init__(self, syllable_vocab, max_input_len=30, padding_char='<PAD>'):
        self.max_input_len = max_input_len
        self.stress_index = indexer(pre_fill=(padding_char, '<BOS>', '<EOS>'))
        self.pos_index = indexer(pre_fill=(padding_char, '<BOS>', '<EOS>'))
        self.syllable_index = indexer(pre_fill=(padding_char, '<BOS>', '<EOS>', '<UNK>') + tuple(syllable_vocab))

    def __call__(self, sample):
        word_stress = self._pad_sequence(
            [self.stress_index[x] for x in sample[0].split()], self.stress_index)
        pos_tags = self._pad_sequence(
            [self.pos_index[tag] for tag in sample[1].split()], self.pos_index)
        syllables = self._pad_sequence(
            [self.syllable_index.get(syllable.lower(), self.syllable_index['<UNK>'])
             for syllable in sample[2].split()],
            self.syllable_index)
        beat_stress = [2] + list(map(int, sample[3].split())) + [2]
        while len(word_stress) < self.max_input_len:
            word_stress.append(0)
            pos_tags.append(0)
            syllables.append(0)
            beat_stress.append(2)
        return {'stress': torch.LongTensor(word_stress),
                'pos': torch.LongTensor(pos_tags),
                'syllables': torch.LongTensor(syllables),
                'tags': torch.LongTensor(beat_stress)}

    def _pad_sequence(self, sequence, index):
        return [index['<BOS>']] + sequence + [index['<EOS>']]

    def decode(self, syllables, true, pred):
        if not hasattr(self, 'index2syllable'):
            self.index2syllable = sorted(self.syllable_index, key=self.syllable_index.get)
        syllable_str, tag_str = '', ''
        prev_len = 0
        for i in range(syllables.size(0)):
            syllable = syllables[i]
            if syllable.item() == 0:
                break
            syllable = self.index2syllable[syllable.item()]
            syllable = syllable + ' ' + (' ' * abs(len(syllable) - 3) if len(syllable) < 3 else '')
            syllable_str += syllable
            tag_str += '{}/{}'.format(true[i].item(), pred[i].item()) + ' ' * abs(len(syllable) - 3)
        print('Correct ({}):'.format((true[:i] == pred[:i]).sum() / i:.3f))
        print('{}\n{}\n'.format(tag_str, syllable_str))


class Trainer:
    def __init__(self, model: BiLSTMTagger, train_data: DataLoader,
                 dev_data: DataLoader = None, test_data: DataLoader = None,
                 optimizer=None, loss_fn=None, device=None, decoder=None):
        self.logger = logging.getLogger('Trainer({})'.format(model.__class__.__name__))
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.decoder = decoder
        self.best_loss = np.inf

    def train(self, epochs=10):
        self.logger.info("Working on {}".format(self.device))
        for epoch in range(1, epochs + 1):
            self.logger.info('Starting epoch {}'.format(epoch))
            self._train_batches(epoch, epochs)
            self.logger.info('Finished epoch {}'.format(epoch))
            if self.dev_data is not None:
                self.model.eval() # set all trainable attributes to False
                self._validate(self.dev_data)
                self.model.train() # set all trainable attributes back to True

    def _train_batches(self, epoch, epochs):
        epoch_loss = 0
        for i, batch in enumerate(self.train_data):
            stress = torch.autograd.Variable(batch['stress']).to(self.device)
            pos = torch.autograd.Variable(batch['pos']).to(self.device)
            syllables = torch.autograd.Variable(batch['syllables']).to(self.device)
            targets = torch.autograd.Variable(batch['tags']).to(self.device)
            
            self.model.zero_grad()

            self.model.hidden = self.model.init_hidden(stress.size(0))
            tag_scores = self.model(stress, pos, syllables)

            loss = self.loss_fn(tag_scores.view(-1, tag_scores.size(2)), targets.view(-1))
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss
            
            if i > 0 and i % 50 == 0:
                logging.info(
                    'Epoch [{}/{}], Step [{}/{:.0f}]'.format(
                        epoch + 1, epochs, i, len(data) / batch_size))
                logging.info(
                    'Loss: {}'.format(epoch_loss.item() / i))
    
    def _validate(self, data: DataLoader, test=False):
        self.logger.info('Validating model')
        y_true, y_pred, accuracy = [], [], 0
        run_loss, example_print = 0, 0
        for i, batch in enumerate(data):
            stress = torch.autograd.Variable(batch['stress'])
            pos = torch.autograd.Variable(batch['pos'])
            syllables = torch.autograd.Variable(batch['syllables'])
            targets = torch.autograd.Variable(batch['tags'])
            
            self.model.zero_grad()
            
            self.model.hidden = self.model.init_hidden(stress.size(0))
            tag_scores = self.model(stress, pos, syllables)

            if not test:
                loss = self.loss_fn(tag_scores.view(-1, tag_scores.size(2)), targets.view(-1))
                run_loss += loss

            # collect predictions
            pred = tag_scores.view(-1, tag_scores.size(2)).data.numpy().argmax(1)
            true = targets.view(-1).data.numpy()
            y_true.append(true)
            y_pred.append(pred)
            true = true.reshape(tag_scores.shape[0], stress.size(1))
            pred = pred.reshape(tag_scores.shape[0], stress.size(1))
            accuracy += (true == pred).all(1).sum() / stress.size(0)
            if i % 10 == 0:
                for elt in np.random.randint(0, true.shape[0], size=2):
                    self.decoder.decode(syllables[elt], true[elt], pred[elt])
        if not test:
            logging.info('Validation Loss: {}'.format(run_loss.item() / len(data)))
            self.save_checkpoint(run_loss.item() / len(data) < self.best_loss)
        p, r, f, _ = sklearn.metrics.precision_recall_fscore_support(
            np.hstack(y_true), np.hstack(y_pred))
        for i in (0, 1):
            logging.info('Validation Scores: c={} p={:.3f}, r={:.3f}, f={:.3f}'.format(
                i, p[i], r[i], f[i]))
        logging.info('Accuracy score: {:.3f}'.format(accuracy / len(data)))

    def save_checkpoint(self, is_best, filename='checkpoint.pth.tar'):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')        

    def test(self, data: DataLoader = None, statefile=None):
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
    parser.add_argument('--hid_dim', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--pretrained_embeddings', required=True, type=str)
    parser.add_argument('--dev_size', default=0.05, type=float)
    parser.add_argument('--test_size', default=0.05, type=float)
    args = parser.parse_args()
    
    syllables, syllable_vectors = load_gensim_embeddings(args.pretrained_embeddings)
    transformer = Sample2Tensor(syllables)
    # load the data with the specified transformer
    data = VerseData(args.dataset, transform=transformer, shuffle_data=True)
    # split the data into a training, development and test set
    train, dev, test = data.train_dev_test_split(
        dev_size=args.dev_size, test_size=args.test_size)
    # for each data set, creat a Dataloader object with specifier batch size
    batch_size = args.batch_size
    train_batches = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    dev_batches = torch.utils.data.DataLoader(dev, batch_size=batch_size, shuffle=False)
    test_batches = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    # construct a layer holding the pre-trained word2vec embeddings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
    syllable_encoder = embedding_layer(syllable_vectors).to(device)
    # Initialize the tagger
    tagger = BiLSTMTagger(
        args.emb_dim, args.hid_dim, args.num_layers, args.dropout, 5,
        40, syllable_encoder, 3, batch_size, device)
    # define the loss function. We choose CrossEntropyloss
    loss_function = torch.nn.CrossEntropyLoss()
    # The Adam optimizer seems to be working fine. Make sure to exlcude the pretrained
    # embeddings from optimizing
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, tagger.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay
    ) #RMSprop
    tagger.to(device)
    trainer = Trainer(
        tagger, train_batches, dev_batches, test_batches, optimizer,
        loss_function, decoder=transformer, device=device
    )
    trainer.train(epochs=args.epochs)
    trainer.test(statefile='model_best.pth.tar')
