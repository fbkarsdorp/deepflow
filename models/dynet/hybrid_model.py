

import time
import random

import dynet

from model import RNNLanguageModel


class HybridLanguageModel(RNNLanguageModel):
    """
    Hybrid LanguageModel
    """
    def __init__(self, encoder, layers, input_dim, cemb_dim, hidden_dim, cond_dim,
                 dropout=0.0, word_dropout=0.0, builder=dynet.CoupledLSTMBuilder):

        self.layers = layers
        self.input_dim = input_dim
        self.cemb_dim = cemb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        model = dynet.ParameterCollection()
        wvocab = encoder.word.size()
        cvocab = encoder.char.size()

        # embeddings
        # - word-level
        self.wembeds = model.add_lookup_parameters((wvocab, input_dim))

        # - char-level
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
        self.char_builder = builder(1, cemb_dim + hidden_dim, hidden_dim, model)
        self.W_char = model.add_parameters((cvocab, hidden_dim))
        self.b_char = model.add_parameters((cvocab))

        # persistence
        self.model = model
        self.modelname = self.get_modelname()
        # add all params to the following list in expected order (& modify from_spec)
        self.spec = (encoder, layers, input_dim, cemb_dim, hidden_dim, cond_dim,
                     builder, dropout)

        # training
        self.prev_hidden = None

    @staticmethod
    def from_spec(spec, model):
        """
        Required by dynet for serialization
        """
        (encoder, layers, input_dim, cemb_dim, hidden_dim, cond_dim,
         builder, dropout) = spec
        return RNNLanguageModel(
            encoder, layers, input_dim, cemb_dim, hidden_dim, cond_dim,
            builder, dropout)

    def embed_sequence(self, words, chars, conds):
        """
        Compute sequence of embeddings for an input sentence
        """
        embs = [dynet.concatenate([self.wembeds[w], self.char_embed(c)])
                for w, c in zip(words, chars)]

        # add conditional embeddings
        if conds:
            conds = [self.conds[c][conds[c]] for c in sorted(self.conds)]
            embs = [dynet.concatenate([emb, *conds]) for emb in embs]

        return embs

    def loss(self, words, chars, conds, test=False):
        # embeddings
        embs = self.embed_sequence(words[:-1], chars[:-1], conds)
        if not test:
            embs = [dynet.dropout(emb, self.dropout) for emb in embs]
            self.builder.set_dropout(self.dropout)
            self.char_builder.set_dropout(self.dropout)
        else:
            self.builder.disable_dropout()
            self.char_builder.disable_dropout()

        W_char, b_char = self.W_char, self.b_char

        state = self.builder.initial_state(self.get_prev_hidden())
        cstate = self.char_builder.initial_state()

        losses = []
        for i, wstate in enumerate(state.add_inputs(embs)):
            wout = wstate.output()
            ctarget = chars[i+1]
            for j, cinp in enumerate(ctarget[:-1]):
                cstate = cstate.add_input(dynet.concatenate([self.cembeds[cinp], wout]))
                logits = b_char + (W_char * cstate.output())
                losses.append(dynet.pickneglogsoftmax(logits, ctarget[j + 1]))

        return losses, wstate

    def sample(self, encoder, nwords=20, conds=None, initial_state=None):
        """
        Generate a number of characters
        """
        dynet.renew_cg()
        # data
        inp = encoder.word.bos
        output = []
        # parameters
        state = self.builder.initial_state(initial_state)
        cstate = self.char_builder.initial_state()
        # sample conditions if needed
        conds = conds or {}
        for c in self.conds:
            # sample conds
            if c not in conds:
                conds[c] = random.choice(list(encoder.conds[c].w2i.values()))
        cs = [self.conds[c][conds[c]] for c in sorted(self.conds)]
        # output
        b_char, W_char = self.b_char, self.W_char

        while True:
            # embedding
            emb = self.wembeds[inp]
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
                coutput.append(encoder.char.i2w[cinp])

            # prepare next word input
            inp = encoder.word.transform_item(''.join(coutput))

            # slighly inefficient that we learn to generate <eos> character by character
            if inp == encoder.word.eos:
                break
            if nwords and len(output) >= nwords:
                break

            output.append(''.join(coutput))

        conds = {c: encoder.conds[c].i2w[cond] for c, cond in conds.items()}

        return ' '.join(output), conds


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
    parser.add_argument('--maxsize', type=int, default=10000)
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
    lm = HybridLanguageModel(encoder, args.layers, args.input_dim, args.cemb_dim,
                             args.hidden_dim, args.cond_dim, dropout=args.dropout)

    print("Storing model to path {}".format(lm.modelname))

    print("Training model")
    lm.train(train, encoder, epochs=args.epochs, dev=list(dev), lr=args.lr,
             trainer=args.trainer, clipping=args.clipping, minibatch=args.minibatch,
             repfreq=args.repfreq, checkfreq=args.checkfreq, lr_weight=args.lr_weight)
