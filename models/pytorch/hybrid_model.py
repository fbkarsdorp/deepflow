
import random
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import utils
from model import RNNLanguageModel


class HybridLanguageModel(RNNLanguageModel):
    def __init__(self, encoder, layers, wemb_dim, cemb_dim, hidden_dim, cond_dim,
                 dropout=0.0):

        self.layers = layers
        self.wemb_dim = wemb_dim
        self.cemb_dim = cemb_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.dropout = dropout
        self.modelname = self.get_modelname()
        super().__init__()

        wvocab = encoder.word.size()
        cvocab = encoder.char.size()

        cnll_weight = torch.ones(wvocab)
        cnll_weight[encoder.char.pad] = 0
        self.register_buffer('cnll_weight', cnll_weight)

        # embeddings
        self.wembs = nn.Embedding(wvocab, wemb_dim, padding_idx=encoder.word.pad)
        self.cembs = nn.Embedding(cvocab, cemb_dim, padding_idx=encoder.char.pad)
        self.cembs_rnn = nn.LSTM(cemb_dim, cemb_dim//2, bidirectional=True)
        input_dim = wemb_dim + cemb_dim

        # conds
        self.conds = {}
        for cond, cenc in encoder.conds.items():
            cemb = nn.Embedding(cenc.size(), cond_dim)
            self.add_module('cond_{}'.format(cond), cemb)
            self.conds[cond] = cemb
            input_dim += cond_dim

        # rnn
        self.rnn = nn.LSTM(input_dim, hidden_dim, layers, dropout=dropout)

        # output
        self.cout_rnn = nn.LSTM(cemb_dim + hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, cvocab)

        self.init()

    def init(self):
        pass

    def forward(self, word, nwords, char, nchars, conds, hidden=None):
        # - embeddings
        # (seq x batch x wemb_dim)
        wembs = self.wembs(word)
        # (seq x batch x cemb_dim)
        cembs = self.embed_chars(char, nchars, nwords)
        embs = torch.cat([wembs, cembs], -1)

        # - conditions
        if conds:
            conds = [self.conds[c](conds[c]) for c in sorted(conds)]
            # expand
            seq, batch = word.size()
            conds = [c.expand(seq, batch, -1) for c in conds]
            # concatenate
            embs = torch.cat([embs, *conds], -1)

        embs = F.dropout(embs, p=self.dropout, training=self.training)

        # - rnn
        embs, unsort = utils.pack_sort(embs, nwords)
        outs, hidden = self.rnn(embs, hidden)
        outs, _ = unpack(outs)
        outs = outs[:, unsort]
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, unsort], hidden[1][:, unsort]
        else:
            hidden = hidden[:, unsort]

        # - compute char-level logits
        # (nwords x hidden_dim)
        outs = utils.flatten_padded_batch(outs, nwords)
        # (nchars x nwords x cemb_dim + hidden_dim)
        cinp = torch.concatenate([self.cembs(char), outs.expand(len(char), -1, -1)], -1)
        # run rnn
        cinp, unsort = utils.pack_sort(cinp, nchars)
        # (nchars x nwords x hidden_dim)
        couts, _ = self.cout_rnn(cinp)
        couts, _ = unpack(couts)
        couts = couts[:, unsort]
        # logits: (nchars x nwords x vocab)
        logits = self.proj(couts)

        return logits, hidden

    def loss(self, logits, word, nwords, char, nchars):
        # - remove <l> tokens from targets and </l> tokens from inputs
        breaks = list(itertools.accumulate(nwords))
        # indices to remove <l> tokens from targets
        index = [i for i in range(sum(nwords)) if i not in breaks][1:]
        index = torch.tensor(index).to(self.device)
        # (nchars x nwords - batch x vocab)
        nchars, _, vocab = logits.size()
        logits = logits.gather(1, index[:, None, :].expand(nchars, len(index), vocab))
        # indices to remove </l> from the input
        index = [i for i in range(sum(nwords)) if i+1 not in breaks]
        index = torch.tensor(index).to(self.device)
        # (nchars x nwords - batch)
        targets = char.gather(1, index.repeat(len(char), 1))

        # - remove </w> from char logits and <w> from char targets
        logits, targets = logits[:-1], targets[1:]
        loss = F.cross_entropy(
            logits.view(-1, vocab), targets.view(-1),
            weight=self.nll_weight, size_average=False)

        # remove 1 char per word instance and 1 char per sentence (<l> tokens)
        insts = sum(nchars) - len(nchars) - len(nwords)

        return loss, insts

    def sample(self, encoder, nsyms=100, conds=None, hidden=None):
        """
        Generate stuff
        """
        # TODO: batch sampling
        batch = 1

        # sample conditions if needed
        conds, bconds = conds or {}, []
        for c in sorted(self.conds):
            # sample conds
            if c not in conds:
                conds[c] = random.choice(list(encoder.conds[c].w2i.values()))
            # compute embedding
            bcond = torch.tensor([conds[c]] * batch, dtype=torch.int64).to(self.device)
            bcond = self.conds[c](bcond)
            bconds.append(bcond.expand(1, batch, -1))

        word = [encoder.word.bos] * batch  # (batch)
        word = torch.tensor(word, dtype=torch.int64).to(self.device)
        char, nchars = [[encoder.char.bol] * batch], [1] * batch  # (1 x nwords)
        char = torch.tensor(char, dtype=torch.int64).to(self.device)

        output = []

        with torch.no_grad():
            for _ in range(nsyms):

                if word[0].item() == encoder.word.eos:
                    break

                # embeddings
                wemb = self.wembs(word.unsqueeze(0))
                cemb = self.embed_chars(char, nchars, [1] * batch)
                embs = torch.cat([wemb, cemb], -1)
                if conds:
                    embs = torch.cat([embs, *bconds], -1)

                outs, hidden = self.rnn(embs, hidden)
                # (1 x 1 x vocab)
                logits = self.proj(outs)
                # (1 x 1 x vocab) -> (1 x vocab)
                logits = logits.squeeze(0)

                preds = F.log_softmax(logits, dim=-1)
                word = (preds / 1).exp().multinomial(1)
                word = word.squeeze(0)

                # accumulate
                output.append(word.tolist())

                # get character-level input
                char, nchars = [], []
                for w in output[-1]:
                    w = encoder.word.i2w[w]
                    c = encoder.char.transform(w)
                    char.append(c)
                    nchars.append(len(c))
                char = torch.tensor(char, dtype=torch.int64).to(self.device).t()

        conds = {c: encoder.conds[c].i2w[cond] for c, cond in conds.items()}
        output = [step[0] for step in output]  # single-batch for now
        output = ' '.join([encoder.word.i2w[step] for step in output])

        return output, conds

from utils import CorpusEncoder, CorpusReader
import utils
reader = CorpusReader('./data/mcflow/mcflow-train.jsonl')
encoder = CorpusEncoder.from_corpus(reader)
batches = reader.get_batches(5)
sents, conds = next(batches)
(w,nw),(c,nc),cs=encoder.transform_batch(sents, conds)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--wemb_dim', type=int, default=100)
    parser.add_argument('--cemb_dim', type=int, default=100)
    parser.add_argument('--cond_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=250)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--maxsize', type=int, default=10000)
    parser.add_argument('--disable_chars', action='store_true')
    # train
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_weight', type=float, default=1.0)
    parser.add_argument('--trainer', default='Adam')
    parser.add_argument('--clipping', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--word_dropout', type=float, default=0.2)
    parser.add_argument('--minibatch', type=int, default=20)
    parser.add_argument('--repfreq', type=int, default=1000)
    parser.add_argument('--checkfreq', type=int, default=0)
    # pytorch
    parser.add_argument('--device', default='cpu')
    # extra
    parser.add_argument('--penn', action='store_true')

    args = parser.parse_args()

    from utils import CorpusEncoder, CorpusReader

    print("Encoding corpus")
    start = time.time()
    train = CorpusReader(args.train)
    dev = CorpusReader(args.dev)

    encoder = CorpusEncoder.from_corpus(train, most_common=args.maxsize)
    print("... took {} secs".format(time.time() - start))

    print("Building model")
    lm = HybridLanguageModel(encoder, args.layers, args.wemb_dim, args.cemb_dim,
                             args.hidden_dim, args.cond_dim, dropout=args.dropout)
    print(lm)
    print("Storing model to path {}".format(lm.modelname))
    lm.to(args.device)

    print("Training model")
    lm.train_model(train, encoder, epochs=args.epochs, minibatch=args.minibatch,
                   dev=dev, lr=args.lr, trainer=args.trainer, clipping=args.clipping,
                   repfreq=args.repfreq, checkfreq=args.checkfreq,
                   lr_weight=args.lr_weight, bptt=1)
