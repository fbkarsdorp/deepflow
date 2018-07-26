
import math
import time
import random

import tqdm
import dynet


class RNNLanguageModel:
    """
    RNNLanguageModel

    use_chars : bool, whether to use char-level embeddings
    tie_weights : bool, tie input and output embeddings (requires same `input_dim`
        and `hidden_dim`)
    """
    def __init__(self, encoder, layers, input_dim, cemb_dim, hidden_dim, cond_dim,
                 use_chars=False, dropout=0.0, word_dropout=0.0,
                 builder=dynet.CoupledLSTMBuilder, tie_weights=False):

        self.use_chars = use_chars
        self.layers = layers
        self.input_dim = input_dim
        self.cemb_dim = cemb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.tie_weights = tie_weights
        model = dynet.ParameterCollection()
        wvocab = encoder.word.size()
        cvocab = encoder.char.size()

        # embeddings
        # - word-level
        if self.tie_weights and input_dim != hidden_dim:
            self.tie_weights = False
            print("Cannot tie weights, input_dim {} != hidden_dim {}".format(
                input_dim, hidden_dim))
        if self.tie_weights:
            self.wembeds = model.add_parameters((wvocab, input_dim))
        else:
            self.wembeds = model.add_lookup_parameters((wvocab, input_dim))

        # - char-level
        if use_chars:
            self.cembeds = model.add_lookup_parameters((cvocab, cemb_dim))
            self.fchars = builder(1, cemb_dim, cemb_dim, model)
            self.bchars = builder(1, cemb_dim, cemb_dim, model)
            input_dim += (2 * cemb_dim)

        # - conditions
        self.conds = {cond: model.add_lookup_parameters((enc.size(), cond_dim))
                      for cond, enc in encoder.conds.items()}
        input_dim += cond_dim * len(encoder.conds)

        # rnn
        self.builder = builder(layers, input_dim, hidden_dim, model)

        # output
        self.W = None
        self.bias = model.add_parameters((wvocab))
        if self.tie_weights:
            self.W = self.wembeds
        else:
            self.W = model.add_parameters((wvocab, hidden_dim))

        # persistence
        self.model = model
        self.modelname = self.get_modelname()
        # add all params to the following list in expected order (& modify from_spec)
        self.spec = (encoder, layers, input_dim, cemb_dim, hidden_dim, cond_dim,
                     builder, use_chars, dropout, tie_weights)

        # training
        self.prev_hidden = None

    def get_modelname(self):
        from datetime import datetime

        return "{}.{}".format(
            type(self).__name__,
            datetime.now().strftime("%Y-%m-%d+%H:%M:%S"))

    def param_collection(self):
        """
        Required by dynet for serialization
        """
        return self.model

    @staticmethod
    def from_spec(spec, model):
        """
        Required by dynet for serialization
        """
        (encoder, layers, input_dim, cemb_dim, hidden_dim, cond_dim,
         builder, use_chars, dropout, tie_weights) = spec
        return RNNLanguageModel(
            encoder, layers, input_dim, cemb_dim, hidden_dim, cond_dim,
            builder, use_chars, dropout, tie_weights)

    def save(self, fpath):
        """
        Save to path
        """
        dynet.save(fpath, [self])

    @staticmethod
    def from_path(fpath):
        """
        Load model from path
        """
        model = dynet.load(fpath, dynet.ParameterCollection())
        return model

    def char_embed(self, chars):
        """
        Compute char-level embeddings with BiLSTM over the characters of a word
        """
        inp = [self.cembeds[c] for c in chars]
        fcembs = self.fchars.initial_state().transduce(inp)
        bcembs = self.bchars.initial_state().transduce(reversed(inp))
        return dynet.concatenate([fcembs[-1], bcembs[-1]])

    def word_embed(self, word):
        """
        Compute word embedding for a single word
        """
        if self.tie_weights:
            return dynet.pick(self.wembeds, index=word, dim=0)
        else:
            return self.wembeds[word]

    def embed_sequence(self, words, chars, conds):
        """
        Compute sequence of embeddings for an input sentence
        """
        embs = [self.word_embed(w) for w in words]

        # add char-level embeddings
        if self.use_chars:
            embs = [dynet.concatenate([emb, self.char_embed(w)])
                    for emb, w in zip(embs, chars)]

        # add conditional embeddings
        if conds:
            conds = [self.conds[c][conds[c]] for c in sorted(self.conds)]
            embs = [dynet.concatenate([emb, *conds]) for emb in embs]

        return embs

    def reset_hidden(self, state, reset=True):
        """
        Reset the hidden state for the next sequence
        """
        if reset:
            self.prev_hidden = None
        else:
            self.prev_hidden = [h.npvalue() for h in state.s()]

    def get_prev_hidden(self):
        """
        Prepare the hidden state from previous sequence to be used as initial
        state of the current sequence
        """
        if self.prev_hidden is None:
            return None
        else:
            return [dynet.inputTensor(hidden) for hidden in self.prev_hidden]

    def loss(self, words, chars, conds, test=False):
        """
        Loss for a single input sentence
        """
        # embeddings
        embs = self.embed_sequence(words[:-1], chars[:-1], conds)
        if not test:
            embs = [dynet.dropout(emb, self.dropout) for emb in embs]

        # rnn
        if test:
            self.builder.disable_dropout()
        else:
            self.builder.set_dropout(self.dropout)
        state = self.builder.initial_state(self.get_prev_hidden())

        bias, W = self.bias, self.W

        # run rnn and compute per-word loss
        losses = []
        for i, state in enumerate(state.add_inputs(embs)):
            logits = bias + (W * state.output())
            losses.append(dynet.pickneglogsoftmax(logits, words[i+1]))

        return losses, state

    def sample(self, encoder, nchars=100, conds=None, initial_state=None):
        """
        Generate a number of syllables
        """
        dynet.renew_cg()
        # data
        inp = encoder.word.bos
        output = []
        # parameters
        state = self.builder.initial_state(initial_state)
        # sample conditions if needed
        conds = conds or {}
        for c in self.conds:
            # sample conds
            if c not in conds:
                conds[c] = random.choice(list(encoder.conds[c].w2i.values()))
        conds = [self.conds[c][conds[c]] for c in sorted(self.conds)]
        # bias, W = dynet.parameter(self.bias), dynet.parameter(self.W)
        bias, W = self.bias, self.W

        while True:
            # embedding
            emb = self.word_embed(inp)
            if self.use_chars:
                if inp == encoder.word.bos or inp == encoder.word.eos:
                    cinp = encoder.char_dummy
                else:
                    cinp = encoder.char.transform(encoder.word.i2w[inp])
                emb = dynet.concatenate([emb, self.char_embed(cinp)])
            if conds:
                emb = dynet.concatenate([emb, *conds])

            # rnn
            state = state.add_input(emb)

            # sample
            probs = dynet.softmax(bias + (W * state.output())).vec_value()
            rand = random.random()
            for inp, prob in enumerate(probs):
                rand -= prob
                if rand <= 0:
                    break

            if inp == encoder.word.eos:
                break
            if nchars and len(output) > nchars:
                break

            output.append(inp)

        conds = {c: encoder.conds[c].i2w[cond] for c, cond in conds.items()}
        output = ' '.join([encoder.word.i2w[i] for i in output])

        return output, conds

    def dev(self, corpus, encoder, best_loss, fails):
        """
        Run dev check
        """
        titems = tloss = 0
        for sent, conds, reset in tqdm.tqdm(corpus):
            dynet.renew_cg()
            (word, char), conds = encoder.transform(sent, conds)
            losses, state = self.loss(word, char, conds, test=True)
            self.reset_hidden(state, reset)
            tloss += dynet.esum(losses).scalar_value()
            titems += len(losses)

        tloss = math.exp(tloss / titems)
        print("Dev loss: {:g}".format(tloss))

        if tloss < best_loss:
            print("New best dev loss: {:g}".format(tloss))
            best_loss = tloss
            fails = 0
            self.save(self.modelname)
        else:
            fails += 1
            print("Failed {} time to improve best dev loss: {}".format(fails, best_loss))

        print()
        for _ in range(20):
            print(self.sample(encoder))
        print()

        return best_loss, fails

    def train(self, corpus, encoder, epochs=5, lr=0.001, shuffle=False,
              clipping=5, dev=None, patience=3, minibatch=15, trainer='Adam',
              repfreq=1000, checkfreq=0, lr_weight=1):
        """
        Train model a number of epochs

        corpus : Iterable over tuples of (sent, conds, reset), sent is a list of strings,
            conds a dictionary holding sentence metadata (conditions), and reset is a
            boolean indicating whether the current line is the first in a new song
            (used to reset the hidden state of the RNN)
        encoder : instance of CorpusEncoder
        shuffle : whether to shuffle the corpus after each epoch. Not to be used with
            very large corpora since it requires putting the corpus in memory
        clipping : max clipping value of the gradient
        dev : instance of CorpusEncoder corresponding to the dev corpus
        patience : for early stopping
        minibatch : number of sentences per minibatch
        trainer : trainer type
        repfreq : number of sentences to wait until reporting train loss
        checkfreq : number of sentence to wait until running an evaluation check
            (0 means only one is run after full epoch)
        """
        if trainer.lower() == 'adam':
            trainer = dynet.AdamTrainer(self.model, lr)
        elif trainer.lower() == 'sgd':
            trainer = dynet.SimpleSGDTrainer(self.model, learning_rate=lr)
        elif trainer.lower() == 'momentum':
            trainer = dynet.MomentumSGDTrainer(self.model, learning_rate=lr)
        else:
            raise ValueError("Unknown trainer: {}".format(trainer))
        trainer.set_clip_threshold(clipping)

        best_loss, fails = float('inf'), 0

        for e in range(epochs):
            if shuffle:
                corpus = list(corpus)
                random.shuffle(corpus)

            tinsts = tloss = 0.0
            binsts, bloss = 0, []
            start = time.time()

            for idx, (sent, conds, reset) in enumerate(corpus):
                # exit if early stopping
                if fails >= patience:
                    print("Early stopping after {} steps".format(fails))
                    print("Best dev loss {:g}".format(best_loss))
                    return

                (word, char), conds = encoder.transform(sent, conds)
                losses, state = self.loss(word, char, conds)
                # never store hidden when shuffling
                self.reset_hidden(state, reset=reset or shuffle)
                bloss.extend(losses)
                binsts += 1

                if binsts == minibatch:
                    loss = dynet.esum(bloss) / len(bloss)
                    loss.backward()
                    trainer.update()
                    tloss += dynet.esum(bloss).scalar_value()
                    tinsts += len(bloss)
                    dynet.renew_cg()
                    bloss, binsts = [], 0

                # report stuff
                if idx and idx % repfreq == 0:
                    speed = int(tinsts / (time.time() - start))
                    print("Epoch {:<3} items={:<10} loss={:<10g} items/sec={}"
                          .format(e, idx, math.exp(min(tloss/tinsts, 100)), speed))
                    tinsts = tloss = 0.0
                    start = time.time()

                if dev and checkfreq and idx and idx % checkfreq == 0:
                    best_loss, fails = self.dev(dev, encoder, best_loss, fails)
                    bloss = []

            if dev and not checkfreq:
                best_loss, fails = self.dev(dev, encoder, best_loss, fails)

            # update lr
            lr = lr * lr_weight
            trainer.restart(lr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--input_dim', type=int, default=100)
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
    # dynet
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-gpus')
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-seed')
    # extra
    parser.add_argument('--penn', action='store_true')

    args = parser.parse_args()

    from utils import CorpusEncoder, CorpusReader, PennReader

    print("Encoding corpus")
    start = time.time()
    if args.penn:
        train = PennReader(args.train)
        dev = PennReader(args.dev)
    else:
        train = CorpusReader(args.train)
        dev = CorpusReader(args.dev)

    encoder = CorpusEncoder.from_corpus(train, most_common=args.maxsize)
    print("... took {} secs".format(time.time() - start))

    print("Building model")
    lm = RNNLanguageModel(encoder, args.layers, args.input_dim, args.cemb_dim,
                          args.hidden_dim, args.cond_dim,
                          use_chars=not args.disable_chars,
                          dropout=args.dropout, tie_weights=args.tie_weights)

    print("Storing model to path {}".format(lm.modelname))

    print("Training model")
    lm.train(train, encoder, epochs=args.epochs, dev=list(dev), lr=args.lr,
             trainer=args.trainer, clipping=args.clipping, minibatch=args.minibatch,
             repfreq=args.repfreq, checkfreq=args.checkfreq, lr_weight=args.lr_weight)
