
import collections
import ijson

import torch


def get_rhymes_from_line(line):
    c, rhyme_type, start, done = -1, '', None, False

    for s in line:
        rhyme = s['rhyme']

        c += 1

        if rhyme == '.' and not rhyme_type:
            continue

        if not rhyme_type:
            start = c

        rhyme_type += rhyme

        if rhyme_type.startswith('(') and rhyme.endswith(')') or \
           rhyme_type.startswith('[') and rhyme.endswith(']') or \
           rhyme.isalpha():
            done = True

        if rhyme_type and done:
            yield rhyme_type, (start, c)
            rhyme_type, start, done = '', None, False


def match_rhymes(line, rhyme):
    c, start = -1, None
    current, *rest = rhyme

    for s in line:
        c += 1
        for r in s['rhyme']:

            if r == '.':    # ignore "." rhyme syllables
                continue

            if r == current:
                if start is not None:  # inside match
                    if not rest:  # done
                        yield start, c
                        start = None
                        current, *rest = rhyme  # reset
                    else:                       # advance position
                        current, *rest = ''.join(rest)
                else:       # starting match
                    start = c
                    if not rest:  # single step rhyme
                        yield start, c
                        start = None
                    else:
                        current, *rest = ''.join(rest)

            else:           # rule stops matching
                if start is not None:
                    start = None
                    current, *rest = rhyme


def get_rhyme_pairs_from_verse(verse):
    for i, this in enumerate(verse):
        sources = collections.defaultdict(list)
        for rh, pos in get_rhymes_from_line(this):
            sources[rh].append(pos)

        for rh, sources in sources.items():
            for j, that in enumerate(verse):
                if i >= j:
                    continue

                targets = list(match_rhymes(that, rh.replace('.', '')))
                if targets:
                    yield rh, (i, sources), (j, targets)


def load_mcflow_lines(path='./data/mcflow/mcflow-large.json'):
    with open(path) as f:
        for song in ijson.items(f, 'item'):
            for verse in song['text']:
                yield from verse
                yield None


def line_to_syllables(line):
    for w in line:
        for idx in range(len(w['syllables'])):
            yield {'syllable': w['syllables'][idx],
                   'start': idx == 0,
                   'end': idx == len(w['syllables']) - 1,
                   'wb': idx == len(w['syllables']),
                   'rhyme': w['rhyme'][idx],
                   'stress': w['stress'][idx],
                   'word': w['word'],
                   'ipa': w['ipa'][idx]}


def load_mc_verses_from_lines(**kwargs):
    """
    import itertools; verse = list(itertools.islice(load_mc_verses_from_lines(), 50))[-1]
    """
    verse = []
    for line in load_mcflow_lines(**kwargs):
        if line is not None:
            verse.append(list(line_to_syllables(line)))
        else:
            yield verse
            verse = []


def display_pairs(pair, verse):
    import termcolor
    rhyme, (one, sources), (two, targets) = pair
    line1, line2 = verse[one], verse[two]

    l1 = ['{:<10}'.format(s['syllable']) for s in line1]
    for start, end in sources:
        for idx, s in enumerate(l1):
            if idx >= start and idx <= end:
                l1[idx] = termcolor.colored(l1[idx], 'green')

    l2 = ['{:<10}'.format(s['syllable']) for s in line2]
    for (start, end) in targets:
        for idx, s in enumerate(l2):
            if idx >= start and idx <= end:
                l2[idx] = termcolor.colored(l2[idx], 'green')

    l1, l2 = ''.join(l1), ''.join(l2)

    print(rhyme, one, two)
    print(l1)
    print(l2)


def get_syllable_features(syl):
    from ipapy.ipastring import IPAString
    import unicodedata

    if syl['ipa'] == 'R':       # represents silence syllables
        return None

    try:
        ipa = IPAString(unicode_string=syl['ipa'].replace('I', 'ɪ'))
    except ValueError:
        # manually fix some errors
        ipa = IPAString(unicode_string={'Nis': 'nis',
                                        'ɾoU': 'ɾou',
                                        'Vin': 'vin',
                                        'vIN': 'vɪn',
                                        'ɾe-': 'ɾe'}[syl['ipa']])

    start, end, idx = None, None, 0
    for ph in ipa:
        if ph.is_vowel:
            if start is None:
                start = idx
            if end is not None:
                raise ValueError("Discontinued nucleus in: {}".format(str(ph)))
        elif ph.is_consonant:
            if start is not None and end is None:
                end = idx

        idx += len([c for c in ph.unicode_repr if unicodedata.category(c) != 'Mn'])

    if start is None:
        assert end is None
        return '', syl['ipa'], ''

    onset = syl['ipa'][:start]
    nucleus = syl['ipa'][start:end]
    if end is not None:
        coda = syl['ipa'][end:]
    else:
        coda = ''

    return onset, nucleus, coda


def get_syllable_vocab(**kwargs):
    vocab = {'onset': collections.Counter(),
             'nucleus': collections.Counter(),
             'coda': collections.Counter()}

    for verse in load_mc_verses_from_lines(**kwargs):
        for line in verse:
            for syl in line:
                if syl['ipa'] == 'R':
                    continue
                onset, nucleus, coda = get_syllable_features(syl)
                vocab['onset'][onset] += 1
                vocab['nucleus'][nucleus] += 1
                vocab['coda'][coda] += 1

    return vocab


def preprocess_line(line):
    inp = []
    for syl in line:
        onset, nucleus, coda = get_syllable_features(syl) or ('', '', '')

        syllable = syl['syllable']
        if syl['start'] and syl['end']:
            pass
        elif syl['start']:
            syllable = syllable + '-'
        elif syl['end']:
            syllable = '-' + syllable
        else:
            syllable = '-' + syllable + '-'

        inp.append({
            'onset': [onset],
            'nucleus': [nucleus],
            'coda': [coda],
            'word': [syl['word']],
            'syllable': [syllable],
            'stress': [syl['stress']]})
    return inp


def fit_encoders():
    from models.loaders import Encoder
    names = ['stress', 'onset', 'nucleus', 'coda', 'syllable', 'word']
    encoders = {name: Encoder(name, eos_token=None, bos_token=None) for name in names}

    for verse in load_mc_verses_from_lines():
        for line in verse:
            for encoder in encoders.values():
                encoder.transform(preprocess_line(line))

    for encoder in encoders.values():
        encoder.fixed_vocab = True

    return encoders


def get_rhyme_mask(sources, targets, slen, tlen):
    mat = torch.zeros((slen, tlen))
    for start1, end1 in sources:
        for start2, end2 in targets:
            mat[start1:end1+1, start2:end2+1] = 1

    return mat


def make_batch(a, alen, b, blen, alignment, pad, device):
    alen_max, blen_max, batch = max(alen), max(blen), len(alignment)

    alignment_b = torch.zeros(batch, alen_max, blen_max).fill_(pad)

    a_b, b_b = {}, {}
    for field in a[0]:
        a_b[field] = torch.zeros(
            alen_max, batch, dtype=torch.int64).fill_(pad)
        b_b[field] = torch.zeros(
            blen_max, batch, dtype=torch.int64).fill_(pad)

    for i in range(batch):
        alen_i, blen_i = alen[i], blen[i]
        alignment_b[i, 0:alen_i, 0:blen_i].copy_(alignment[i])
        for field in a[0]:
            a_b[field][0:alen_i, i].copy_(a[i][field])
            b_b[field][0:blen_i, i].copy_(b[i][field])

    alignment_b = alignment_b.to(device)
    for field in a_b:
        a_b[field] = a_b[field].to(device)
        b_b[field] = b_b[field].to(device)

    return a_b, list(alen), b_b, list(blen), alignment_b


def get_verse_data(verse, encoders):
    # transform input into vectors
    tverse, lens = [], []
    for line in verse:
        tverse.append({field: encoder.transform(preprocess_line(line))
                       for field, encoder in encoders.items()})
        lens.append(len(line))

    for _, (i, sources), (j, targets) in get_rhyme_pairs_from_verse(verse):
        alignment = get_rhyme_mask(sources, targets, lens[i], lens[j])
        yield tverse[i], lens[i], tverse[j], lens[j], alignment


def get_iterator(dataset, batch_size, pad, device='cpu'):
    import random
    random.shuffle(dataset)
    for i in range(0, len(dataset), batch_size):
        a, alen, b, blen, alignment = zip(*dataset[i: i+batch_size])
        yield make_batch(a, alen, b, blen, alignment, pad, device)


def get_dataset(encoders, **kwargs):
    examples = []
    for verse in load_mc_verses_from_lines(**kwargs):
        examples.extend(list(get_verse_data(verse, encoders)))

    return examples
