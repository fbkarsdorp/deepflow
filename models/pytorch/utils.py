
import json
import collections

import torch
import pronouncing


BOS, EOS, BOL, EOL, UNK, PAD = '<s>', '</s>', '<l>', '</l>', '<unk>', '<pad>'


def bucket_length(length, buckets=(5, 10, 15, 20)):
    for i in sorted(buckets, reverse=True):
        if length >= i:
            return i
    return min(buckets)


class Vocab:
    def __init__(self, counter, most_common=1e+6, **reserved):
        self.w2i = {}
        for key, sym in reserved.items():
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


def get_batch(sents, pad, device):
    lengths = [len(sent) for sent in sents]
    batch, maxlen = len(sents), max(lengths)
    t = torch.zeros(batch, maxlen, dtype=torch.int64) + pad
    for idx, (sent, length) in enumerate(zip(sents, lengths)):
        t[idx, :length].copy_(torch.tensor(sent))

    t = t.t().contiguous().to(device)

    return t, lengths


class CorpusEncoder:
    def __init__(self, word, conds):
        self.word = word
        c2i = collections.Counter(c for w in word.w2i for c in w)
        self.char = Vocab(c2i, eos=EOS, bos=BOS, unk=UNK, pad=PAD, eol=EOL, bol=BOL)
        self.conds = conds

    @classmethod
    def from_corpus(cls, corpus, most_common=25000):
        w2i = collections.Counter()
        conds_w2i = collections.defaultdict(collections.Counter)
        for sent, conds, *_ in corpus:
            for cond in conds:
                conds_w2i[cond][conds[cond]] += 1

            for word in sent:
                w2i[word] += 1

        word = Vocab(w2i, bos=BOS, eos=EOS, unk=UNK, pad=PAD, most_common=most_common)
        conds = {c: Vocab(cond_w2i) for c, cond_w2i in conds_w2i.items()}

        return cls(word, conds)

    def transform_batch(self, sents, conds, device='cpu'):  # conds is a list of dicts
        # word-level batch
        words, nwords = get_batch(
            [self.word.transform(s) for s in sents], self.word.pad, device)

        # char-level batch
        chars = []
        for sent in sents:
            sent = [self.char.transform(w) for w in sent]
            sent = [[self.char.bol]] + sent + [[self.char.eol]]
            chars.extend(sent)
        chars, nchars = get_batch(chars, self.char.pad, device)

        # conds
        bconds = {}
        for c in self.conds:
            batch = torch.tensor([self.conds[c].transform_item(d[c]) for d in conds])
            batch = batch.to(device)
            bconds[c] = batch

        return (words, nwords), (chars, nchars), bconds


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

    def lines_from_jsonl(self, path):
        with open(path, errors='ignore') as f:
            for idx, line in enumerate(f):
                try:
                    for verse in json.loads(line)['text']:
                        for line in verse:
                            sent, conds = self.prepare_line(line)
                            if len(sent) >= 2:  # avoid too short sentences for LM
                                yield sent, conds
                except json.decoder.JSONDecodeError:
                    print("Couldn't read song #{}".format(idx+1))

    def __iter__(self):
        yield from self.lines_from_jsonl(self.fpath)

    def get_batches(self, batch_size):
        songs = []
        with open(self.fpath, errors='ignore') as f:
            for idx, line in enumerate(f):
                if len(songs) >= batch_size:
                    # yield
                    for batch in zip(*songs):  # implicitely cuts down to minlen
                        sents, conds = zip(*batch)
                        yield sents, conds
                    # reset
                    songs = []
                try:
                    song = json.loads(line)['text']
                    songs.append([self.prepare_line(l) for verse in song for l in verse])
                except json.decoder.JSONDecodeError:
                    print("Couldn't read song #{}".format(idx+1))


class PennReader:
    def __init__(self, fpath):
        self.fpath = fpath

    def __iter__(self):
        with open(self.fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line.split(), {}, False


def pack_sort(inp, lengths, batch_first=False):
    """
    Transform input into PaddedSequence sorting batch by length (as required).
    Also return an index variable that unsorts the output back to the original
    order.

    Parameters:
    -----------
    inp: torch.Tensor(seq_len x batch x dim)
    lengths: LongTensor of length ``batch``

    >>> from torch.nn.utils.rnn import pad_packed_sequence as unpack
    >>> inp = torch.tensor([[1, 3], [2, 4], [0, 5]], dtype=torch.float)
    >>> lengths = torch.tensor([2, 3]) # unsorted order
    >>> sorted_inp, unsort = pack_sort(inp, lengths)
    >>> sorted_inp, _ = unpack(sorted_inp)
    >>> sorted_inp[:, unsort].tolist()  # original order
    [[1.0, 3.0], [2.0, 4.0], [0.0, 5.0]]
    >>> sorted_inp.tolist()  # sorted by length
    [[3.0, 1.0], [4.0, 2.0], [5.0, 0.0]]
    """
    from torch.nn.utils.rnn import pack_padded_sequence

    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths)  # no need to use gpu

    lengths, sort = torch.sort(lengths, descending=True)
    _, unsort = sort.sort()

    if batch_first:
        inp = pack_padded_sequence(inp[sort], lengths.tolist())
    else:
        inp = pack_padded_sequence(inp[:, sort], lengths.tolist())

    return inp, unsort


def pad_flat_batch(emb, nwords, maxlen):
    """
    Transform a 2D flat batch (batch of words in multiple sentences) into a 3D
    padded batch where words have been allocated to their respective sentence
    according to user passed sentence lengths `nwords`

    Parameters
    ===========
    emb : torch.Tensor(total_words x emb_dim), flattened tensor of word embeddings
    nwords : torch.Tensor(batch), number of words per sentence

    Returns
    =======
    torch.Tensor(max_seq_len x batch x emb_dim) where:
        - max_seq_len = max(nwords)
        - batch = len(nwords)

    >>> emb = [[0], [1], [2], [3], [4], [5]]
    >>> nwords = [3, 1, 2]
    >>> pad_flat_batch(torch.tensor(emb), torch.tensor(nwords)).tolist()
    [[[0], [3], [4]], [[1], [0], [5]], [[2], [0], [0]]]
    """
    with torch.no_grad():
        if len(emb) != sum(nwords):
            raise ValueError("Got {} items but was asked to pad {}"
                             .format(len(emb), sum(nwords)))

        output, last = [], 0

        for sentlen in nwords:
            padding = (0, 0, 0, maxlen - sentlen)
            output.append(torch.nn.functional.pad(emb[last:last+sentlen], padding))
            last = last + sentlen

        # (seq_len x batch x emb_dim)
        output = torch.stack(output, dim=1)

    return output


def flatten_padded_batch(batch, nwords):
    """
    Inverse of pad_flat_batch

    Parameters
    ===========
    batch : tensor(seq_len, batch, encoding_size), output of the encoder
    nwords : tensor(batch), lengths of the sequence (without padding)

    Returns
    ========
    tensor(nwords, encoding_size)

    >>> batch = [[[0], [3], [4]], [[1], [0], [5]], [[2], [0], [0]]]
    >>> nwords = [3, 1, 2]
    >>> flatten_padded_batch(torch.tensor(batch), torch.tensor(nwords)).tolist()
    [[0], [1], [2], [3], [4], [5]]
    """
    output = []
    for sent, sentlen in zip(batch.t(), nwords):
        output.extend(list(sent[:sentlen].chunk(sentlen)))

    return torch.cat(output, dim=0)


def lines_from_jsonl(path):
    """
    lines = []
    c = 0
    for line, reset in lines_from_jsonl('./data/ohhla-beatstress.jsonl'):
        c += 1
        if reset:
            lines.append(c)
            c = 0
    """
    import json

    reset = True
    with open(path, errors='ignore') as f:
        for idx, line in enumerate(f):
            try:
                for verse in json.loads(line)['text']:
                    for line in verse:
                        yield line, reset
                        reset = False
                    reset = True
            except json.decoder.JSONDecodeError:
                print("Couldn't read song #{}".format(idx+1))


def get_consecutive_rhyme_pairs(path):
    """
    rhymes=list(get_consecutive_rhyme_pairs('./data/ohhla-beatstress.jsonl'))
    Couldn't read song #21768
    Couldn't read song #32470
    Couldn't read song #38108
    Couldn't read song #38581

    In [49]: len(rhymes)
    Out[49]: 4225689

    In [50]: sum(rhymes)/len(rhymes)
    Out[50]: 0.08086988891042383

    """
    prev = None
    for line, reset in lines_from_jsonl(path):
        if prev is not None and not reset:
            if line[-1]['token'] in pronouncing.rhymes(prev[-1]['token']):
                yield 1
            else:
                yield 0
        else:
            yield 0

        prev = line
