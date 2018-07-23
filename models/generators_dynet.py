
import math
import time
import random
import collections

import tqdm
import dynet


BOS, EOS, UNK = '<s>', '</s>', '<unk>'


def bucket_length(length, buckets=(5, 10, 15, 20)):
    for i in sorted(buckets, reverse=True):
        if length >= i:
            return i
    return min(buckets) 


class Vocab:
    def __init__(self, counter, most_common=1e+6, bos=None, eos=None, unk=None):
        self.w2i = {}
        self.reserved = {'bos': bos, 'eos': eos, 'unk': unk}
        for key, sym in self.reserved.items():
            if sym is not None:
                if sym in counter:
                    print("Removing {} [{}] from training corpus".format(key, sym))
                    del counter[sym]
                self.w2i.setdefault(sym, len(self.w2i))
            setattr(self, key, self.w2i.get(sym))

        for sym, _ in counter.most_common(int(most_common)):
            self.w2i.setdefault(sym, len(self.w2i))
        self.i2w = {i: w for w, i in self.w2i.items()}

    def size(self):
        return len(self.w2i.keys())

    def transform_item(self, item):
        try:
            return self.w2i[item]
        except KeyError:
            if self.unk is None:
                raise ValueError("Couldn't retrieve <unk> for unknown token")
            else:
                return self.unk

    def transform(self, inp):
        out = [self.transform_item(i) for i in inp]
        if self.bos is not None:
            out = [self.bos] + out
        if self.eos is not None:
            out = out + [self.eos]
        return out

    def __getitem__(self, item):
        return self.w2i[item]


class CorpusEncoder:
    def __init__(self, word, conds):
        self.word = word
        c2i = collections.Counter(c for w in word.w2i for c in w)
        self.char = Vocab(c2i, eos=EOS, bos=BOS, unk=UNK)
        self.char_dummy = [self.char.bos, self.char.eos]
        self.conds = conds

    @classmethod
    def from_corpus(cls, corpus, most_common=25000):
        w2i = collections.Counter()
        conds_w2i = collections.defaultdict(collections.Counter)
        for sent, conds, _ in corpus:
            for cond in conds:
                conds_w2i[cond][conds[cond]] += 1

            for word in sent:
                w2i[word] += 1

        word = Vocab(w2i, bos=BOS, eos=EOS, unk=UNK, most_common=most_common)
        conds = {c: Vocab(cond_w2i) for c, cond_w2i in conds_w2i.items()}

        return cls(word, conds)

    def transform(self, sent, conds):
        word = self.word.transform(sent)
        char = [self.char.transform(w) for w in sent]
        char = [self.char_dummy] + char + [self.char_dummy]
        assert len(word) == len(char)
        conds = {c: self.conds[c].transform_item(i) for c, i in conds.items()}

        return (word, char), conds


class CorpusReader:
    def __init__(self, fpath):
        self.fpath = fpath

    def prepare_line(self, line):
        def format_syllables(syllables):
            if len(syllables) == 1:
                return syllables

            output = []
            for idx, syl in enumerate(syllables):
                if idx == 0:
                    output.append(syl + '-')
                elif idx == (len(syllables) - 1):
                    output.append('-' + syl)
                else:
                    output.append('-' + syl + '-')

            return output

        sent = [syl for w in line for syl in format_syllables(w['syllables'])]
        # TODO: fill this with other conditions such as:
        # - rhyme (encode last rhyming sequence of syllables if they are found in
        #          the dictionary of rhymes---see load_rhymes.py from master, or
        #          just the last word otherwise)
        conds = {'length': bucket_length(len(sent))}

        return sent, conds

    def lines_from_json(self, path):
        import ijson
        reset = False
        with open(path) as f:
            for song in ijson.items(f, 'item'):
                for verse in song['text']:
                    for line in verse:
                        sent, conds = self.prepare_line(line)
                        if len(sent) >= 2:  # avoid too short sentences for LM
                            yield sent, conds, reset
                            reset = False
                reset = True

    def lines_from_jsonl(self, path):
        import json
        reset = False
        with open(path, errors='ignore') as f:
            for idx, line in enumerate(f):
                try:
                    for verse in json.loads(line)['text']:
                        for line in verse:
                            sent, conds = self.prepare_line(line)
                            if len(sent) >= 2:  # avoid too short sentences for LM
                                yield sent, conds, reset
                                reset = False
                    reset = True
                except json.decoder.JSONDecodeError:
                    print("Couldn't read song #{}".format(idx+1))
                    reset = True

    def __iter__(self):
        if self.fpath.endswith('jsonl'):
            yield from self.lines_from_jsonl(self.fpath)
        else:
            yield from self.lines_from_json(self.fpath)


class PennReader:
    def __init__(self, fpath):
        self.fpath = fpath

    def __iter__(self):
        with open(self.fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line.split(), {}, False


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

        # # for char-level loss the following needs to be added
        # self.char_builder = builder(1, cemb_dim + hidden_dim, hidden_dim, model)
        # self.W_char = model.add_parameters((cvocab, hidden_dim))
        # self.b_char = model.add_parameters((cvocab))

        # persistence
        self.model = model
        # add all params to the following list in expected order (& modify from_spec)
        self.spec = (encoder, layers, input_dim, cemb_dim, hidden_dim, cond_dim,
                     builder, use_chars, dropout, tie_weights)

        # training
        self.prev_hidden = None

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

        if self.use_chars:
            cembs = [self.char_embed(w) for w in chars]
            embs = [dynet.concatenate([wemb, cemb]) for wemb, cemb in zip(embs, cembs)]
        if conds:
            cs = [self.conds[c][conds[c]] for c in sorted(self.conds)]
            embs = [dynet.concatenate([emb, *cs]) for emb in embs]
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
        # # you might need to uncomment next line if dynet complains
        # bias, W = dynet.parameter(self.bias), dynet.parameter(self.W)

        # run rnn and compute per-syllable loss
        losses = []
        for i, state in enumerate(state.add_inputs(embs)):
            logits = bias + (W * state.output())
            losses.append(dynet.pickneglogsoftmax(logits, words[i+1]))

        return dynet.esum(losses), state

    def sample(self, encoder, nchars=100, conds=None):
        """
        Generate a number of syllables
        """
        dynet.renew_cg()
        # data
        inp = encoder.word.bos
        output = []
        # parameters
        state = self.builder.initial_state()
        # sample conditions if needed
        conds = conds or {}
        for c in self.conds:
            # sample conds
            if c not in conds:
                conds[c] = random.choice(list(encoder.conds[c].w2i.values()))
        cs = [self.conds[c][conds[c]] for c in sorted(self.conds)]
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
                emb = dynet.concatenate([emb, *cs])

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

    # TODO: char-level loss needs implementing:
    #   - self.char_builder (RNN with )
    #   - char-level embs for the loss RNN (or reuse from self.cembeds input embs?)
    #   - self.char_W, self.char_b (output layer for char-level loss)
    def char_level_loss(self, words, chars, conds, test=False):
        # embeddings
        embs = self.embed_sequence(words[:-1], chars[:-1], conds)
        if not test:
            embs = [dynet.dropout(emb, self.dropout) for emb in embs]
            self.builder.set_dropout(self.dropout)
            self.char_builder.set_dropout(self.dropout)
        else:
            self.builder.disable_dropout()
            self.char_builder.disable_dropout()

        W_char, b_char = self.W_char, b_char

        losses, cstate = [], self.char_builder.initial_state()
        for i, wstate in enumerate(state.add_inputs(embs)):
            wout, loss = wstate.output(), []
            for j, cinp in enumerate(chars[i+1][:-1]):
                cstate = cstate.add_input(dynet.concatenate([self.cembeds[cinp], wout]))
                logits = b_char + (W_char * cstate.output())
                loss.append(dynet.pickneglogsoftmax(logits, chars[i+1][j+1]))
            losses.append(dynet.esum(loss) / (len(chars[i+1]) - 1))

        return dynet.esum(losses)

    def sample_char_level(self, encoder, nwords=20, conds=None):
        """
        Generate a number of characters
        """
        dynet.renew_cg()
        # data
        inp = encoder.word.bos
        output = []
        # parameters
        state = self.builder.initial_state()
        # sample conditions if needed
        conds = conds or {}
        for c in self.conds:
            # sample conds
            if c not in conds:
                conds[c] = random.choice(list(encoder.conds[c].w2i.values()))
        cs = [self.conds[c][conds[c]] for c in sorted(self.conds)]
        # bias, W = dynet.parameter(self.bias), dynet.parameter(self.W)
        bias, W = self.bias, self.W
        b_char, W_char = self.b_char, self.W_char

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
                emb = dynet.concatenate([emb, *cs])

            # rnn
            state = state.add_input(emb)

            # char-level
            cinp = encoder.char.bos
            cstate = self.char_builder.initial_state()
            coutput = []
            while True:
                cemb = dynet.concatenate([self.cembeds[cinp], state.output()])
                cstate = cstate.add_input(cemb)
                probs = dynet.softmax(b_char + (W_char * cstate.output())).vec_value()
                rand = random.random()
                for cinp, prob in enumerate(probs):
                    rand -= prob
                    if rand <= 0:
                        break
                if cinp == encoder.char.eos:
                    break
                coutput.append(cinp)

            # prepare next word input
            inp = encoder.word.transform_item(
                ''.join(encoder.char.w2i[c] for c in coutput))

            # slighly inefficient that we learn to generate <eos> character by character
            if inp == encoder.word.eos:
                break
            if nwords and len(output) > nwords:
                break
            output.append(inp)

        conds = {c: encoder.conds[c].i2w[cond] for c, cond in conds.items()}
        output = ' '.join([encoder.word.i2w[i] for i in output])

        return output, conds

    def dev(self, corpus, encoder, best_loss, fails):
        """
        Run dev check
        """
        dynet.renew_cg()

        items = tloss = 0
        for sent, conds, reset in tqdm.tqdm(corpus):
            (word, char), conds = encoder.transform(sent, conds)
            loss, state = self.loss(word, char, conds, test=True)
            self.reset_hidden(state, reset)
            tloss += loss.scalar_value()
            items += len(sent)

        tloss = math.exp(tloss / items)
        print("Dev loss: {:g}".format(tloss))

        if tloss < best_loss:
            print("New best dev loss: {:g}".format(tloss))
            best_loss = tloss
            fails = 0
            self.save("generator")
        else:
            fails += 1
            print("Failed {} time to improve best dev loss: {}".format(fails, best_loss))

        print()
        for _ in range(5):
            print(self.sample(encoder))
        print()

        return best_loss, fails

    def train(self, corpus, encoder, epochs=5, lr=0.001, shuffle=False,
              clipping=5, dev=None, patience=3, minibatch=15, trainer='Adam',
              repfreq=1000, checkfreq=0):
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

            items = tloss = 0.0
            bloss = []
            start = time.time()

            for idx, (sent, conds, reset) in enumerate(corpus):
                # exit if early stopping
                if fails >= patience:
                    print("Early stopping after {} steps".format(fails))
                    print("Best dev loss {:g}".format(best_loss))
                    return
                
                (word, char), conds = encoder.transform(sent, conds)
                loss, state = self.loss(word, char, conds)
                reset = reset or shuffle  # never store hidden when shuffling
                self.reset_hidden(state, reset)
                bloss.append(loss)

                if len(bloss) == minibatch:
                    loss = dynet.esum(bloss)
                    (loss/len(bloss)).backward()  # backprop batch average loss
                    trainer.update()
                    tloss += loss.scalar_value()
                    dynet.renew_cg()
                    bloss = []

                items += len(sent)

                # report stuff
                if idx and idx % repfreq == 0:
                    speed = int(items / (time.time() - start))
                    print("Epoch {:<3} items={:<10} loss={:<10g} items/sec={}"
                          .format(e, idx, math.exp(min(tloss/items, 100)), speed))
                    items = tloss = 0.0
                    start = time.time()

                if dev and checkfreq and idx and idx % checkfreq == 0:
                    best_loss, fails = self.dev(dev, encoder, best_loss, fails)
                    bloss = []

            if dev and not checkfreq:
                best_loss, fails = self.dev(dev, encoder, best_loss, fails)


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
    # train
    parser.add_argument('--lr', type=float, default=0.001)
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
                          args.hidden_dim, args.cond_dim, use_chars=True,
                          dropout=args.dropout, tie_weights=args.tie_weights)

    print("Training model")
    lm.train(train, encoder, epochs=args.epochs, dev=list(dev), lr=args.lr,
             trainer=args.trainer, clipping=args.clipping, minibatch=args.minibatch,
             repfreq=args.repfreq, checkfreq=args.checkfreq)
