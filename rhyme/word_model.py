import collections
import random
from typing import Tuple, List, Set, Dict

import pronouncing
import sklearn.model_selection
import torch

import numpy as np
random.seed(1999)
np.random.seed(1999)
torch.manual_seed(1999)


class RhymeData:
    def __init__(self, pad_symbol='<PAD>', neg_example_rate=1):
        self.neg_example_rate = neg_example_rate
        self.pairs, self.chars = self._compile_data()
        source, target, y = zip(*self.pairs)
        X = list(zip(source, target))
        X_train, X_dev, y_train, y_dev = sklearn.model_selection.train_test_split(
            X, y, stratify=y, test_size=0.1, shuffle=True)
        self.target_ratio = sum(i == 1 for i in y_train) / len(y_train)
        source, target = zip(*X_train)
        self.train = list(zip(source, target, y_train))
        source, target = zip(*X_dev)
        self.dev = list(zip(source, target, y_dev))
        print(len(self.train))
        self.index = collections.defaultdict()
        self.index.default_factory = lambda: len(self.index)
        self._pad = self.index[pad_symbol]
        self._bow = self.index['<BOW>']
        self._eow = self.index['<EOW>']

    def get_train_batches(self, batch_size=20):
        return self._get_batches(self.train, batch_size=batch_size)

    def get_dev_batches(self, batch_size=20):
        return self._get_batches(self.dev, batch_size=batch_size)

    def _get_batches(self, data, batch_size=20):
        n, max_len = 0, 0
        in0, in1, targets = [], [], []
        for x1, x2, y in data:
            in0.append([self._bow] + [self.index[c] for c in x1] + [self._eow])
            if len(in0[-1]) > max_len:
                max_len = len(in0[-1])
            in1.append([self._bow] + [self.index[c] for c in x2] + [self._eow])
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
        clean_word = lambda word: ''.join(c for c in word if c.isalpha())
        pronouncing.init_cmu()
        chars = set()
        vocab = sorted(map(clean_word, pronouncing.lookup.keys()))
        self.vocab = random.sample(vocab, 20000)
        suffix_index = collections.defaultdict(set)
        for word in pronouncing.lookup.keys():
            for suffix_len in (3, 4, 5):
                suffix_index[word[-min(suffix_len, len(word)):]].add(word)
        pos_pairs, neg_pairs = set(), set()
        for j, word in enumerate(self.vocab):
            assert word
            rhymes_set = set(pronouncing.rhymes(word))
            rhymes = list(rhymes_set)
            if rhymes:
                for rhyme in rhymes:
                    pos_pairs.add((word, rhyme, 1))
                    chars.update(set(list(word) + list(rhyme)))
                non_rhymes, neighbors = set(), set()
                for suffix_len in (3, 4, 5):
                    neighbors = neighbors.union(suffix_index[word[-min(suffix_len, len(word)):]])
                neighbors = list(neighbors)
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if len(non_rhymes) >= (len(rhymes) / 2):
                        break
                    if neighbor != word and neighbor not in rhymes_set:
                        non_rhymes.add(neighbor)
                while len(non_rhymes) < len(rhymes):
                    non_rhyme = random.sample(vocab, 1)[0]
                    if non_rhyme not in rhymes_set and non_rhyme != word:
                        non_rhymes.add(non_rhyme)
                assert len(non_rhymes) == len(rhymes)
                for negative in non_rhymes:
                    neg_pairs.add((word, negative, 0))
                    chars.update(set(list(negative)))
        return list(pos_pairs.union(neg_pairs)), chars

    def decode(self, inputs):
        if not hasattr(self, 'index2char'):
            self.index2char = sorted(self.index, key=self.index.get)
        return ''.join(self.index2char[c] for c in inputs
                       if c not in (self._pad, self._bow, self._eow))


class SigmoidObjective(torch.nn.Module):
    def __init__(self, encoding_size: int) -> None:
        super(SigmoidObjective, self).__init__()
        self.logits = torch.nn.Linear(encoding_size, 1, bias=False)

    def forward(self, enc1, enc2, p=0.0):
        logits = (enc1 - enc2) ** 2
        logits = torch.nn.functional.dropout(logits, p=p, training=self.training)
        return self.logits(logits).squeeze(1)

    def loss(self, pred, labels):
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, labels)

    def score(self, output):
        return torch.nn.functional.sigmoid(output)

    def predict(self, score):
        return score > 0.5    


class CauchyObjective(torch.nn.Module):
    def __init__(self, encoding_size):
        super(CauchyObjective, self).__init__()
        self.logits = torch.nn.Linear(encoding_size, 1, bias=False)

    def forward(self, enc1, enc2, p=0.0):
        logits = (enc1 - enc2) ** 2
        logits = torch.nn.functional.dropout(logits, p=p, training=self.training)
        return 1 / (1 + self.logits(logits).clamp(min=0).squeeze(1))

    def loss(self, pred, labels):
        return torch.nn.functional.binary_cross_entropy(pred, labels)

    def score(self, output):
        return output

    def predict(self, score):
        return score > 0.5


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, weight=0.5, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.weight = weight
        self.margin = margin

    def forward(self, x0, x1):
        return torch.nn.functional.cosine_similarity(x0, x1)

    def loss(self, sims, y):
        pos = self.weight * ((1 - sims) ** 2)
        neg = (sims * torch.gt(sims, self.margin).float()) ** 2
        return torch.mean(y * pos + (1 - y) * neg)

    def score(self, output):
        return output

    def predict(self, score):
        return score > 0.5


class SiameseLSTM(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers, dropout):
        super(SiameseLSTM, self).__init__()

        self.char_embeddings = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            emb_dim, hid_dim // 2, n_layers, batch_first=True, bidirectional=True)
        self.projection_layer = torch.nn.Linear(hid_dim, vocab_size)
        self.dropout = dropout

    def forward_once(self, inputs):
        embs = self.char_embeddings(inputs)
        embs = torch.nn.functional.dropout(embs, p=self.dropout, training=self.training)
        outputs, _ = self.lstm(embs)
        return outputs[:,-1,:]

    def forward(self, x0, x1):
        output0 = self.forward_once(x0)
        output1 = self.forward_once(x1)
        return output0, output1


def get_rhymes(word: str, objective: str, n: int = 20, batch_size=200) -> List[Tuple[str, int]]:
    rhymes = []
    word = [data._bow] + [data.index[c] for c in word if c in data.index] + [data._eow]
    targets, stargets = [], []
    lens = [len(word)]
    for starget in data.vocab:
        target = [data._bow] + [data.index[c] for c in starget if c in data.index] + [data._eow]
        targets.append(target)
        stargets.append(starget)
        lens.append(len(target))
        if len(targets) == batch_size:
            max_len = max(lens)
            target = torch.LongTensor([[0] * (max_len - len(t)) + t for t in targets]).to(device)
            qword = torch.LongTensor([[0] * (max_len - len(word)) + word]).to(device) 
            with torch.no_grad():       
                out1, out2 = siamese(qword, target)
                scores = objective(out1, out2)
                preds = objective.predict(scores)
                for i, pred in enumerate(preds.tolist()):
                    if pred == 1:
                        rhymes.append((stargets[i], scores[i].item()))
            targets, stargets, lens = [], [], [len(word)]
    rhymes.sort(key=lambda i: i[1], reverse=True)
    return rhymes[:n]


if __name__ == '__main__':
    data = RhymeData(neg_example_rate=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siamese = SiameseLSTM(30, 200, len(data.chars) + 3, 3, 0.5).to(device)
    print(data.target_ratio)
    loss_fn = ContrastiveLoss(weight=data.target_ratio, margin=-0.5)
    optimizer = torch.optim.Adam(siamese.parameters(), lr=0.001)
    for epoch in range(1000):
        epoch_loss = 0
        for i, batch in enumerate(data.get_train_batches(batch_size=500), 1):
            x0, x1, y = batch
            optimizer.zero_grad()
            out1, out2 = siamese(x0.to(device), x1.to(device))
            pred = loss_fn(out1, out2)
            loss = loss_fn.loss(pred, y.to(device))
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            if i % 50 == 0:
                print(f'loss in epoch {epoch}/{i} is {epoch_loss / i}')
        siamese.eval()
        loss_fn.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        for i, batch in enumerate(data.get_dev_batches(batch_size=200), 1):
            with torch.no_grad():            
                x0, x1, y = batch
                out1, out2 = siamese(x0.to(device), x1.to(device))
                pred = loss_fn(out1, out2)
                loss = loss_fn.loss(pred, y.to(device))
                val_loss += loss.item()
                preds = loss_fn.predict(pred)
                all_targets.extend(y.tolist())
                all_preds.extend(preds.tolist())
        testword = random.sample(data.vocab, 1)[0]
        print(f'Testing for {testword}')
        rhymes = get_rhymes(testword, loss_fn)
        print(rhymes)
        print(f'   val loss after epoch {epoch} is {val_loss / i}')
        print(sklearn.metrics.classification_report(all_targets, all_preds))
        print(sklearn.metrics.accuracy_score(all_targets, all_preds))
        siamese.train()
        loss_fn.train()
