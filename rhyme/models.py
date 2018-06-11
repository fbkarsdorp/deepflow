import collections
import random

import pronouncing
import sklearn.model_selection
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.nn.utils.rnn import pad_sequence


class RhymeData:
    def __init__(self, pad_symbol='<PAD>', neg_example_rate=1):
        self.neg_example_rate = neg_example_rate
        self.pairs = self._compile_data()
        self.train, _ = sklearn.model_selection.train_test_split(
            self.pairs, test_size=0.1, shuffle=True)
        self.train, self.dev = sklearn.model_selection.train_test_split(
            self.train, test_size=0.1, shuffle=True)
        print(len(self.train))
        self.index = collections.defaultdict()
        self.index.default_factory = lambda: len(self.index)
        self.index[pad_symbol]

    def get_train_batches(self, batch_size=200):
        return self._get_batches(self.train, batch_size=batch_size)

    def get_dev_batches(self, batch_size=200):
        return self._get_batches(self.dev, batch_size=batch_size)

    def _get_batches(self, data, batch_size=20):
        n, max_len = 0, 0
        in0, in1, targets = [], [], []
        for x1, x2, y in data:
            in0.append([self.index[c] for c in x1])
            if len(in0[-1]) > max_len:
                max_len = len(in0[-1])
            in1.append([self.index[c] for c in x2])
            if len(in1[-1]) > max_len:
                max_len = len(in1[-1])
            targets.append(y)
            n += 1
            if n == batch_size:
                x0, x1, y = self.post_batch(in0, in1, targets, max_len)
                yield x0, x1, y
                in0, in1, targets = [], [], []
                n, max_len = 0, 0
        if in0:
            x0, x1, y = self.post_batch(in0, in1, targets, max_len)
            yield x0, x1, y

    def post_batch(self, in0, in1, targets, max_len):
        targets = torch.FloatTensor(targets)
        in0 = torch.LongTensor([[0] * (max_len - len(x)) + x for x in in0])
        in1 = torch.LongTensor([[0] * (max_len - len(x)) + x for x in in1])
        return in0, in1, targets
        
    def _compile_data(self):
        pronouncing.init_cmu()
        self.vocab = vocab = random.sample(sorted(pronouncing.lookup.keys()), 100)
        vectorizer = CountVectorizer(token_pattern=r'\w')
        D = vectorizer.fit_transform(vocab)
        DM = pairwise_distances(D, metric='cosine')
        pairs = []
        for j, word in enumerate(vocab):
            word = ''.join(c for c in word if c.isalpha())
            if not word:
                continue
            rhymes = pronouncing.rhymes(word)
            assert isinstance(rhymes, list)
            if rhymes:
                for rhyme in rhymes:
                    if random.random() > 0.5:
                        pairs.append((word, rhyme, 1))
                    else:
                        pairs.append((rhyme, word, 1))
                non_rhymes = []
                neighbors = DM[j].argsort()                
                while True:
                    if len(non_rhymes) >= len(rhymes) * self.neg_example_rate:
                        break
                    for i in neighbors:
                        if vocab[i] != word and vocab[i] not in rhymes:
                            non_rhymes.append(vocab[i])
                while len(non_rhymes) < len(rhymes) * self.neg_example_rate:
                    non_rhyme = random.sample(vocab, 1)[0]
                    if non_rhyme not in rhymes:
                        non_rhymes.append(non_rhyme)
                for negative in non_rhymes:
                    if random.random() > 0.5:
                        pairs.append((word, negative, 0))
                    else:
                        pairs.append((negative, word, 0))
        return pairs

    def decode(self, inputs):
        if not hasattr(self, 'index2char'):
            self.index2char = sorted(self.index, key=self.index.get)
        return ''.join(self.index2char[c] for c in inputs if c != 0)
                    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        dist = 1 - torch.nn.functional.cosine_similarity(x0, x1, dim=1)
        pos = 0.5 * (dist ** 2)
        neg = torch.clamp(self.margin - dist, min=0.0) ** 2
        return torch.mean(y * pos + (1 - y) * neg)
    

class SiameseLSTM(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers, dropout):
        super(SiameseLSTM, self).__init__()

        self.char_embeddings = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(emb_dim, hid_dim // 2, n_layers, batch_first=True, bidirectional=True)
        self.projection_layer = torch.nn.Linear(hid_dim, vocab_size)
        self.dropout = dropout

    def forward_once(self, inputs):
        embs = self.char_embeddings(inputs)
        embs = torch.nn.functional.dropout(embs, p=self.dropout, training=self.training)
        outputs, _ = self.lstm(embs)
        return outputs[:,-1,:] # torch.mean(outputs, 1) #

    def forward(self, x0, x1):
        output0 = self.forward_once(x0)
        output1 = self.forward_once(x1)
        return output0, output1


def is_rhyme(word_a, word_b):
    x = torch.LongTensor([[data.index[c] for c in word_a]])
    y = torch.LongTensor([[data.index[c] for c in word_b]])
    with torch.no_grad():
        siamese.eval()
        x, y = siamese(x, y)
        siamese.train()
    return torch.nn.functional.cosine_similarity(x, y) >= 0.5

def get_rhymes(word):
    rhymes = []
    x = torch.LongTensor([[data.index[c] for c in word]])
    for word in data.vocab:
        y = torch.LongTensor([[data.index[c] for c in word]])
        siamese.eval()        
        with torch.no_grad():
            o0, o1 = siamese(x, y)
            if (torch.nn.functional.cosine_similarity(o0, o1) >= 0.5).item() == 1:
                rhymes.append(word)
        siamese.train()
    return rhymes
            

if __name__ == '__main__':
    data = RhymeData()
    siamese = SiameseLSTM(30, 100, 50, 1, 0.5)
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.25)
    optimizer = torch.optim.Adam(siamese.parameters(), lr=0.001)
    for epoch in range(100):
        epoch_loss = 0
        for i, batch in enumerate(data.get_train_batches(), 1):
            x0, x1, y = batch
            optimizer.zero_grad()
            out1, out2 = siamese(x0, x1)
            loss = loss_fn(out1, out2, (2 * y - 1))
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            if i % 50 == 0:
                print(f'loss in epoch {epoch}/{i} is {epoch_loss / i}')
        siamese.eval()
        val_loss = 0
        accuracy = 0
        for i, batch in enumerate(data.get_dev_batches(), 1):
            with torch.no_grad():            
                x0, x1, y = batch
                out1, out2 = siamese(x0, x1)
                loss = loss_fn(out1, out2, (2 * y - 1))
                val_loss += loss.item()
                preds = torch.nn.functional.cosine_similarity(out1, out2) >= 0.5
                for (a, b, c, d) in zip(x0, x1, y, preds):
                    a = data.decode(a)
                    b = data.decode(b)
                    print(a, b, c.item(), d.item())
                accuracy += sklearn.metrics.accuracy_score(y, preds)
        print(f'val loss after epoch {epoch} is {val_loss / i}')
        print(f'accuracy after epoch {epoch} is {accuracy / i}')
        siamese.train()
            
            
            
        
