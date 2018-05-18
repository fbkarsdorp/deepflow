import collections
import random

import gensim
import numpy as np
import sklearn.model_selection
import sklearn.metrics

import torch
import torch.utils.data


random.seed(1685)
np.random.seed(1685)
torch.manual_seed(1685)


def embedding_layer(weights, padding_vector=False, unk_vector=False, trainable=False):
    weights = np.vstack((np.zeros((padding_vector + unk_vector, weights.shape[1])), weights))
    embedding_weights = torch.FloatTensor(weights)
    embedding = torch.nn.Embedding(*embedding_weights.shape)
    embedding.weight = torch.nn.Parameter(embedding_weights)
    embedding.weight.requires_grad = trainable
    return embedding


def load_gensim_embeddings(fpath: str):
    vectors = gensim.models.Word2Vec.load(fpath)
    vectors.init_sims(replace=True)
    vocab = sorted(vectors.wv.vocab.keys())
    vectors = np.array([vectors[w] for w in vocab], dtype=np.float32)
    return vocab, vectors


class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding_dropout_p,
                 stress_size, pos_size, syllable_encoder, tagset_size, batch_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.stress_encoder = torch.nn.Embedding(stress_size, embedding_dim, padding_idx=0)
        self.pos_encoder = torch.nn.Embedding(pos_size, embedding_dim, padding_idx=0)
        self.syllable_encoder = syllable_encoder
        self.dropout = torch.nn.Dropout(p=embedding_dropout_p)
        self.lstm = torch.nn.LSTM(
            embedding_dim * 2 + syllable_encoder.weight.shape[1], hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
                torch.autograd.Variable(
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim)))

    def forward(self, stress_sequence, pos_sequence, syllable_sequence):
        stress_embeddings = self.stress_encoder(stress_sequence)
        pos_embeddings = self.pos_encoder(pos_sequence)
        syllable_embeddings = self.syllable_encoder(syllable_sequence)
        embeddings = self.dropout(torch.cat((stress_embeddings, pos_embeddings, syllable_embeddings), 2))
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores


class BiLSTMTagger(LSTMTagger):
    def __init__(self, embedding_dim, hidden_dim, num_layers, embedding_dropout_p,
                 stress_size, pos_size, syllable_encoder, tagset_size, batch_size):
        super(BiLSTMTagger, self).__init__(
            embedding_dim, hidden_dim, num_layers,
            embedding_dropout_p, stress_size, pos_size, syllable_encoder, tagset_size, batch_size
        )

        self.lstm = torch.nn.LSTM(
            embedding_dim * 2 + syllable_encoder.weight.shape[1], hidden_dim // 2, num_layers=num_layers,
            bidirectional=True, batch_first=True
        )

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(
                    torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2)),
                torch.autograd.Variable(
                    torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
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
        X_train, X_dev = sklearn.model_selection.train_test_split(self.data_samples, test_size=dev_size)
        X_train, X_test = sklearn.model_selection.train_test_split(X_train, test_size=test_size)
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
        self.padding_char = padding_char
        self.item2index = indexer(pre_fill=(padding_char, 0, 1, 2))
        self.syllable_index = indexer(pre_fill=(padding_char, '<UNK>') + tuple(syllable_vocab))

    def __call__(self, sample):
        word_stress = list(map(lambda x: int(x) + 1, sample[0].split()))
        pos_tags = [self.item2index[tag] for tag in sample[1].split()]
        syllables = [self.syllable_index.get(syllable, self.syllable_index['<UNK>']) for syllable in sample[2].split()]
        beat_stress = list(map(int, sample[3].split()))
        while len(word_stress) < self.max_input_len:
            word_stress.append(0)
            pos_tags.append(0)
            syllables.append(0)
            beat_stress.append(0)
        return {'stress': torch.LongTensor(word_stress),
                'pos': torch.LongTensor(pos_tags),
                'syllables': torch.LongTensor(syllables),
                'tags': torch.LongTensor(beat_stress)}

if __name__ == '__main__':
    syllables, syllable_vectors = load_gensim_embeddings('../data/ohhla-syllable-embeddings')
    transformer = Sample2Tensor(syllables)
    batch_size = 30
    data = VerseData('../data/mcflow-syllables.txt', transform=transformer)
    train, dev, test = data.train_dev_test_split(dev_size=0.1, test_size=0.1)
    train_batches = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    dev_batches = torch.utils.data.DataLoader(dev, batch_size=batch_size, shuffle=True)
    test_batches = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    syllable_encoder = embedding_layer(syllable_vectors, padding_vector=True, unk_vector=True)
    tagger = BiLSTMTagger(40, 100, 2, 0.1, 3, 20, syllable_encoder, 2, batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Working on {}".format(device))
    tagger.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, tagger.parameters()), lr=0.001) #RMSprop
    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0
        for i, batch in enumerate(train_batches):
            stress = torch.autograd.Variable(batch['stress']).to(device)
            pos = torch.autograd.Variable(batch['pos']).to(device)
            syllables = torch.autograd.Variable(batch['syllables']).to(device)
            targets = torch.autograd.Variable(batch['tags'])
            tagger.zero_grad()
            tagger.hidden = tagger.init_hidden(stress.size(0))
            tag_scores = tagger(stress, pos, syllables)
            loss = loss_function(tag_scores.view(-1, tag_scores.size(2)), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            if i > 0 and i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(data)/batch_size:.0f}], Loss: {epoch_loss.data[0] / i}\r', end='\r')

    all_preds, all_true = [], []
    accs = 0
    tagger.eval()
    for i, batch in enumerate(test_batches):
        stress = torch.autograd.Variable(batch['stress'])
        pos = torch.autograd.Variable(batch['pos'])
        syllables = torch.autograd.Variable(batch['syllables'])
        targets = torch.autograd.Variable(batch['tags'])
        tagger.zero_grad()
        tagger.hidden = tagger.init_hidden(stress.size(0))
        tag_scores = tagger(stress, pos, syllables)
        preds = tag_scores.view(-1, tag_scores.size(2)).data.numpy().argmax(1)
        y_true = targets.view(-1).data.numpy()
        all_preds.append(preds)
        all_true.append(y_true)
        accs += (y_true.reshape(tag_scores.shape[0], 30) == preds.reshape(tag_scores.shape[0], 30)).sum() / preds.shape[0]
    print(sklearn.metrics.classification_report(np.hstack(all_true), np.hstack(all_preds), digits=4))
    print(accs / len(test_batches))

