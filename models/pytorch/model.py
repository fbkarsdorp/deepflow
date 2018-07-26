
import random
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import tqdm

import utils


class RNNLanguageModel(nn.Module):
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

        nll_weight = torch.ones(wvocab)
        nll_weight[encoder.word.pad] = 0
        self.register_buffer('nll_weight', nll_weight)
        # TODO: pick one of those depending on loss level
        # cnll_weight = torch.ones(wvocab)
        # cnll_weight[encoder.char.pad] = 0
        # self.register_buffer('cnll_weight', cnll_weight)

        # embeddings
        self.wembs = nn.Embedding(wvocab, wemb_dim, padding_idx=encoder.word.pad)
        self.cembs = nn.Embedding(cvocab, cemb_dim, padding_idx=encoder.char.pad)
        self.crnn = nn.LSTM(cemb_dim, cemb_dim//2, bidirectional=True, dropout=dropout)
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
        self.proj = nn.Linear(hidden_dim, wvocab)

        if wemb_dim == hidden_dim:
            print("Tying embedding and projection weights")
            self.proj.weight = self.wembs.weight

        self.init()

    def init(self):
        pass

    def save(self, fpath, encoder):
        self.eval()
        old_device = self.device
        self.to('cpu')
        with open(fpath + ".pt", 'wb') as f:
            torch.save({'model': self, 'encoder': encoder}, f)
        self.train()
        self.to(old_device)

    def get_modelname(self):
        from datetime import datetime

        return "{}.{}".format(
            type(self).__name__,
            datetime.now().strftime("%Y-%m-%d+%H:%M:%S"))

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_chars(self, char, nchars, nwords):
        cembs = self.cembs(char)
        cembs, unsort = utils.pack_sort(cembs, nchars)
        _, hidden = self.crnn(cembs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        cembs = hidden[:, unsort, :].transpose(0, 1).contiguous().view(sum(nwords), -1)
        cembs = utils.pad_flat_batch(cembs, nwords, max(nwords))

        return cembs

    def forward(self, word, nwords, char, nchars, conds, hidden=None):
        # (seq x batch x wemb_dim)
        wembs = self.wembs(word)
        # (seq x batch x cemb_dim)
        cembs = self.embed_chars(char, nchars, nwords)

        embs = torch.cat([wembs, cembs], -1)

        if conds:
            conds = [self.conds[c](conds[c]) for c in sorted(conds)]
            # expand
            seq, batch = word.size()
            conds = [c.expand(seq, batch, -1) for c in conds]
            # concatenate
            embs = torch.cat([embs, *conds], -1)

        embs, unsort = utils.pack_sort(embs, nwords)
        outs, hidden = self.rnn(embs, hidden)
        outs, _ = unpack(outs)
        outs = outs[:, unsort]
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, unsort], hidden[1][:, unsort]
        else:
            hidden = hidden[:, unsort]

        logits = self.proj(outs)

        return logits, hidden

    def loss(self, logits, words, nwords):
        seq, batch, vocab = logits.size()

        loss = F.cross_entropy(
            logits.view(batch * seq, vocab), words.view(-1),
            weight=self.nll_weight, size_average=False)

        # remove 1 per batch instance
        insts = sum(nwords) - len(nwords)

        return loss, insts

    def sample(self, encoder, nsyms=100, conds=None, hidden=None):
        """
        Generate stuff
        """
        self.eval()

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

    def dev(self, corpus, encoder, best_loss, fails):
        hidden = None
        tloss = tinsts = 0

        self.eval()

        with torch.no_grad():
            for sents, conds in tqdm.tqdm(corpus.get_batches(1)):
                (words, nwords), (chars, nchars), conds = encoder.transform_batch(
                    sents, conds, self.device)
                logits, hidden = self(words, nwords, chars, nchars, conds, hidden)
                loss, insts = self.loss(logits[:-1], words[1:], nwords)
                tinsts += insts
                tloss += loss.item()

        self.train()

        tloss = math.exp(tloss / tinsts)
        print("Dev loss: {:g}".format(tloss))

        if tloss < best_loss:
            print("New best dev loss: {:g}".format(tloss))
            best_loss = tloss
            fails = 0
            self.save(self.modelname, encoder)
        else:
            fails += 1
            print("Failed {} time to improve best dev loss: {}".format(fails, best_loss))

        print()
        for _ in range(20):
            print(self.sample(encoder))
        print()

        return best_loss, fails

    def train_model(self, corpus, encoder, epochs=5, lr=0.001, clipping=5, dev=None,
                    patience=3, minibatch=15, trainer='Adam', repfreq=1000, checkfreq=0,
                    lr_weight=1, bptt=1):

        # get trainer
        if trainer.lower() == 'adam':
            trainer = torch.optim.Adam(self.parameters(), lr=lr)
        elif trainer.lower() == 'sgd':
            trainer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError("Unknown trainer: {}".format(trainer))
        print(trainer)

        # local variables
        hidden = None
        best_loss, fails = float('inf'), 0

        for e in range(epochs):

            tinsts = tloss = 0.0
            start = time.time()

            for idx, (sents, conds) in enumerate(corpus.get_batches(minibatch)):
                (words, nwords), (chars, nchars), conds = encoder.transform_batch(
                    sents, conds, self.device)

                # early stopping
                if fails >= patience:
                    print("Early stopping after {} steps".format(fails))
                    print("Best dev loss {:g}".format(best_loss))
                    return

                # forward
                logits, hidden = self(words, nwords, chars, nchars, conds, hidden)

                # loss
                loss, insts = self.loss(logits[:-1], words[1:], nwords)
                (loss/insts).backward(retain_graph=bptt > 1)

                # bptt
                if idx % bptt == 0:
                    # step
                    nn.utils.clip_grad_norm_(self.parameters(), clipping)
                    trainer.step()
                    trainer.zero_grad()
                    # detach
                    if isinstance(hidden, torch.Tensor):
                        hidden = hidden.detach()
                    else:
                        hidden = tuple(h.detach() for h in hidden)

                tinsts, tloss = tinsts + insts, tloss + loss.item()

                if idx and idx % (repfreq // minibatch) == 0:
                    speed = int(tinsts / (time.time() - start))
                    print("Epoch {:<3} items={:<10} loss={:<10g} items/sec={}"
                          .format(e, idx, math.exp(min(tloss/tinsts, 100)), speed))
                    tinsts = tloss = 0.0
                    start = time.time()

                if dev and checkfreq and idx and idx % (checkfreq // minibatch) == 0:
                    best_loss, fails = self.dev(dev, encoder, best_loss, fails)
                    # update lr
                    if fails > 0:
                        for pgroup in trainer.param_groups:
                            pgroup['lr'] = pgroup['lr'] * lr_weight
                        print(trainer)

            if dev and not checkfreq:
                best_loss, fails = self.dev(dev, encoder, best_loss, fails)
                # update lr
                if fails > 0:
                    for pgroup in trainer.param_groups:
                        pgroup['lr'] = pgroup['lr'] * lr_weight
                    print(trainer)


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
    lm = RNNLanguageModel(encoder, args.layers, args.wemb_dim, args.cemb_dim,
                          args.hidden_dim, args.cond_dim, dropout=args.dropout)
    print(lm)
    print("Storing model to path {}".format(lm.modelname))
    lm.to(args.device)

    print("Training model")
    lm.train_model(train, encoder, epochs=args.epochs, minibatch=args.minibatch,
                   dev=dev, lr=args.lr, trainer=args.trainer, clipping=args.clipping,
                   repfreq=args.repfreq, checkfreq=args.checkfreq,
                   lr_weight=args.lr_weight, bptt=1)
