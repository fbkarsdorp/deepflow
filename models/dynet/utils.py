
import collections


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
