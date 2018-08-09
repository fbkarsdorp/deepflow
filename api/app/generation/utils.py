
import re
import json
import collections
import torch
import tqdm

# encoder
BOS, EOS, BOL, EOL, UNK, PAD = '<s>', '</s>', '<l>', '</l>', '<unk>', '<pad>'
# identify punctuation
PUNCT = re.compile(r'[^\w\s]+$')


def detokenize(line, debug=False):
    """
    Detokenize string
    """
    # rules
    PRE       = {'(', '[', '{'}
    POST      = {';', '.', ',', ':', '?', '?!', '!', '!!', '!!!', ')', ']', '}', '...', '..', "'s"}
    AMB       = {'"', "'"}
    CONTS     = {('i', 'ma'), ('i', 'mma'), ('y', 'all'), ('c', 'mon'), ('it', 's'),
                 # french stuff (shouldn't matter too much)
                 ('c', 'qui'), ('c', 'est'), ('n', 'est'), ('j', 'ceux')}
    PRECONTS  = {'gon', 'yo', 'lil'}
    POSTCONTS = {'cause', 'em', 'round', 'till', 'er', 'bout'}

    words = line.split()
    unclosed = set()
    output = ""

    c = 0
    while c < len(words) - 1:
        # prepare input
        if c == 0:
            prev, cur, post = '', words[c], words[c+1]
        else:
            prev, cur, post = words[c-1:c+2]

        # numbers sometimes get tokenized
        if cur.isnumeric() and prev.isnumeric():
            output += cur
        # contractions (r&b)
        elif cur == '&' and prev.lower() == 'r' and post.lower() == 'b':
            output += cur + post
            c += 1
        # contractions on '
        elif cur == "'" and post.lower() == 'until':
            output += ' ' + cur + 'til'
            c += 1
        # triplets like I'ma
        elif cur == "'" and (prev.lower(), post.lower()) in CONTS:
            output += cur + post
            c += 1
        # pre ' contractions
        elif cur == "'" and prev.lower() in PRECONTS:
            output += cur + ' ' + post
            c += 1
        # post ' contractions
        elif cur == "'" and post.lower() in POSTCONTS:
            output += ' ' + cur + post
            c += 1
        # ambiguous quotes (why do they even exist?)
        elif cur in AMB:
            if cur in unclosed:
                # post
                output += cur
                unclosed.remove(cur)
            else:
                # pre
                output += ' ' + cur + post
                unclosed.add(cur)
                c += 1
        # closing stuff
        elif cur in POST:
            output += cur
        # opening stuff
        elif cur in PRE:
            output += ' ' + cur + post
            c += 1
        # normal situation
        else:
            output += ' ' + cur

        c += 1

    output = output.strip()

    # no last bit to add
    if len(words) == 1:
        pass

    # last bit was already added
    elif c == len(words):
        pass

    # finish last bit
    else:
        prev, cur = words[-2:]
        if prev.isnumeric() and cur.isnumeric():
            output += cur
        elif cur in POST:
            output += cur
        elif cur in AMB and cur in unclosed:
            output += cur
        else:
            output += ' ' + cur

    if debug:
        print("line: [{}]".format(line))
        print("outs: [{}]".format(output))

    return output


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


def join_syllables(syllables):
    joint = ''

    for syl in syllables:
        if syl.startswith('-'):
            syl = syl[1:]
        if syl.endswith('-'):
            syl = syl[:-1]
        else:
            syl = syl + ' '
        joint += syl

    return joint


def is_valid(sylls, verbose=False):

    if len(sylls) == 0:
        if verbose:
            print("Empty line")
        return False

    if UNK in sylls:
        if verbose:
            print("Found <unk> in line")
        return False

    if sylls[0].startswith('-'):
        if verbose:
            print("Incongruent start syllabification: {}".format(sylls[0]))
        return False

    if sylls[-1].endswith('-'):
        if verbose:
            print("Incongruent ending syllabification: {}".format(sylls[-1]))
        return False

    for idx in range(len(sylls) - 1):
        if sylls[idx].endswith('-') and not sylls[idx+1].startswith('-'):
            if verbose:
                print("Incongruent syllabification: {} {}".format(
                    sylls[idx], sylls[idx+1]))
            return False

    return True


def is_valid_pair(sylls1, sylls2, verbose=False):

    def get_last(line):
        last = []
        for syl in line[::-1]:
            last.append(syl)
            if not syl.startswith('-'):
                break
        return join_syllables(last[::-1])

    # avoid same word in the end of consecutive lines
    last1, last2 = get_last(sylls1), get_last(sylls2)
    if last1 == last2:
        if verbose:
            print("Lines end equal:\n\t- {}\n\t- {}\n\t".format(last1, last2))
        return False

    return True


def bucket_length(length, buckets=(5, 10, 15, 20)):
    for i in sorted(buckets, reverse=True):
        if length >= i:
            return i
    return min(buckets)


class Vocab:
    def __init__(self, counter, most_common=1e+6, **reserved):
        self.w2i = {}
        self.reserved = {}
        for key, sym in reserved.items():
            if sym in counter:
                print("Removing {} [{}] from training corpus".format(key, sym))
                del counter[sym]
            self.w2i.setdefault(sym, len(self.w2i))
            self.reserved[key] = sym
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

    def to_dict(self):
        return {"reserved": self.reserved,
                'w2i': [{"key": key, "val": val} for key, val in self.w2i.items()]}

    @classmethod
    def from_dict(cls, d):
        inst = cls(collections.Counter())
        inst.w2i = {d["key"]: d["val"] for d in d['w2i']}
        for key, val in d['reserved'].items():
            setattr(inst, key, inst.w2i[val])
        inst.i2w = {val: key for key, val in inst.w2i.items()}

        return inst


def get_batch(sents, pad, device):
    lengths = [len(sent) for sent in sents]
    batch, maxlen = len(sents), max(lengths)
    t = torch.zeros(batch, maxlen, dtype=torch.int64) + pad
    for idx, (sent, length) in enumerate(zip(sents, lengths)):
        t[idx, :length].copy_(torch.tensor(sent))

    t = t.t().contiguous().to(device)

    return t, lengths


class CorpusEncoder:
    def __init__(self, word, char, conds, reverse=False):
        self.word = word
        self.char = char
        self.conds = conds
        self.reverse = reverse

    @classmethod
    def from_corpus(cls, *corpora, most_common=25000, **kwargs):
        # create counters
        w2i = collections.Counter()
        conds_w2i = collections.defaultdict(collections.Counter)
        for sent, conds, *_ in tqdm.tqdm(it for corpus in corpora for it in corpus):
            for cond in conds:
                conds_w2i[cond][conds[cond]] += 1
            for word in sent:
                w2i[word] += 1
        c2i = collections.Counter(c for w in w2i for c in w)

        # create vocabs
        word = Vocab(w2i, most_common=most_common, bos=BOS, eos=EOS, unk=UNK, pad=PAD)
        char = Vocab(c2i, eos=EOS, bos=BOS, unk=UNK, pad=PAD, eol=EOL, bol=BOL, space=' ')
        conds = {c: Vocab(cond_w2i) for c, cond_w2i in conds_w2i.items()}

        return cls(word, char, conds, **kwargs)

    def to_json(self, fpath):
        with open(fpath, 'w') as f:
            json.dump({'word': self.word.to_dict(),
                       'char': self.char.to_dict(),
                       'conds': {cond: c.to_dict() for cond, c in self.conds.items()},
                       'reverse': self.reverse},
                      f)

    @classmethod
    def from_json(cls, fpath):
        with open(fpath) as f:
            obj = json.load(f)

        word = Vocab.from_dict(obj['word'])
        char = Vocab.from_dict(obj['char'])
        conds = {cond: Vocab.from_dict(obj["conds"][cond]) for cond in obj['conds']}

        return cls(word, char, conds, obj['reverse'])

    def transform_batch(self, sents, conds, device='cpu'):  # conds is a list of dicts
        if self.reverse:
            sents = [s[::-1] for s in sents]

        # word-level batch
        words, nwords = get_batch(
            [self.word.transform(s) for s in sents], self.word.pad, device)

        # char-level batch
        chars = []
        for sent in sents:
            sent = [self.char.transform(w) for w in sent]
            sent = [[self.char.bos, self.char.bol, self.char.eos]] + sent
            sent = sent + [[self.char.bos, self.char.eol, self.char.eos]]
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
    def __init__(self, fpath, dpath=None, conds=None):
        self.fpath = fpath
        self.d = None
        if dpath is not None:
            with open(dpath) as f:
                self.d = json.loads(f.read())
        self.conds = conds

    def prepare_line(self, line, prev):
        # prepare line
        sent = []
        for w in line:
            if len(w.get('syllables', [])) == 0:
                if re.match(PUNCT, w['token']):  # this actually always applies
                    sent.append(w['token'])
            else:
                sent.extend(format_syllables(w['syllables']))

        conds = {}

        # get rhyme
        if self.d and not (self.conds is not None and 'rhyme' not in self.conds):
            try:
                # rhyme = get_rhyme2(line, prev, d)
                # if rhyme:
                #     rhyme = '-'.join(rhyme)
                rhyme = get_final_phonology(self.d[line[-1]['token']])
                rhyme = '-'.join(rhyme) if len(rhyme) <= 2 else None
            except KeyError:
                rhyme = None
            conds['rhyme'] = rhyme or UNK

        # get length
        if not (self.conds is not None and 'length' not in self.conds):
            conds['length'] = bucket_length(len(sent))

        return sent, conds

    def lines_from_jsonl(self, path):
        with open(path, errors='ignore') as f:
            for idx, line in enumerate(f):
                try:
                    for verse in json.loads(line)['text']:
                        prev = None
                        for line in verse:
                            sent, conds = self.prepare_line(line, prev)
                            if len(sent) >= 2:  # avoid too short sentences for LM
                                yield sent, conds
                            prev = line
                except json.decoder.JSONDecodeError:
                    print("Couldn't read song #{}".format(idx+1))

    def __iter__(self):
        yield from self.lines_from_jsonl(self.fpath)

    def get_batches(self, batch_size, yield_stops=False):
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
                    if yield_stops:
                        yield None
                try:
                    song = json.loads(line)['text']
                    lines = []
                    for verse in song:
                        prev = None
                        for line in verse:
                            sent, conds = self.prepare_line(line, prev)
                            if len(sent) >= 2:
                                lines.append((sent, conds))
                            prev = line
                    songs.append(lines)
                except json.decoder.JSONDecodeError:
                    print("Couldn't read song #{}".format(idx+1))


def chunks(it, size):
    """
    Chunk a generator into a given size (last chunk might be smaller)
    """
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


class PennReader:
    def __init__(self, fpath, **kwargs):
        self.fpath = fpath

    def __iter__(self):
        with open(self.fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line.split(), {}

    def get_batches(self, batch_size):
        data = chunks(iter(self), batch_size)
        while True:
            try:
                # [[(line1, {}), (line2, {}), ...], ...]
                batches = [next(data) for _ in range(batch_size)]
                # [((line1, {}), (line3, {}), ...), ...]
                batches = list(zip(*batches))
                for batch in batches:
                    sents, conds = zip(*batch)
                    yield sents, conds

            except StopIteration:
                return


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


def get_rhyme(line1, line2, d, return_lines=False):

    def get_vowels(line):
        try:
            phon = d[line[-1]['token'].lower()]  # only last word
            phon = list(filter(lambda ph: ph[-1].isnumeric(), phon.split()))
            return phon
        except KeyError:
            return

    # remove same word rhymes
    if line1[-1]['token'] == line2[-1]['token']:
        return

    vow1, vow2 = get_vowels(line1), get_vowels(line2)
    if not vow1 or not vow2:
        return

    match, done = [], False
    for i in range(min(len(vow1), len(vow2), 3)):
        s1, s2 = vow1[-(i+1)], vow2[-(i+1)]
        match.append((s1, s2))
        if s1.endswith('1') or s2.endswith('1'):
            if s1 == s2:
                done = True
            break

    if not done:
        # didn't find main stress
        return

    if return_lines:
        return [i['token'] for i in line1], [i['token'] for i in line2], match[::-1]
    else:
        return match[::-1]


def get_final_phonology(phon):
    phon = list(filter(lambda ph: ph[-1].isnumeric(), phon.split()))
    rhyme = []
    for ph in phon[::-1]:
        rhyme.append(ph)
        if ph.endswith('1'):
            break

    return rhyme[::-1]


def get_rhyme2(line1, line2, d, return_lines=False):
    """
    This only works with the extra dictionary created by running "add_phon_dict.py"
    """
    last1, last2 = line1[-1]['token'], line2[-1]['token']
    # remove same word rhymes
    if last1 == last2:
        return

    if last2 in d and last1 in d[last2]['rhym']:
        rhyme = get_final_phonology(d[last2]['phon'])

        if return_lines:
            return ([i['token'] for i in line1],
                    [i['token'] for i in line2],
                    rhyme)
        else:
            return rhyme


def get_consecutive_rhyme_pairs_dict(path, dictpath, return_lines=True):
    """
    rhymes = get_consecutive_rhyme_pairs_dict(
        './data/ohhla-beatstress.jsonl', './data/ohhla.vocab.phon.json')
    rhymes = list(rhymes)

    sum(1 for i in rhymes if i)/len(rhymes)
    import collections
    counts=collections.Counter(len(i[-1]) for i in rhymes if i)
    longer=[rhyme for rhyme in rhymes if rhyme and len(rhyme[-1])>25]

    typecount=collections.defaultdict(collections.Counter)
    for rhyme in rhymes:
        if rhyme:
            _,_,rhyme=rhyme
            _,rhyme=zip(*rhyme)
            typecount[len(rhyme)][rhyme]+=1
    """
    with open(dictpath) as f:
        d = json.loads(f.read())

    prev = None
    for line, reset in lines_from_jsonl(path):
        if prev is not None and not reset:
            yield get_rhyme2(line, prev, d, return_lines)
        else:
            yield 0

        prev = line
