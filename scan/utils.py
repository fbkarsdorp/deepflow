import ujson
import numpy as np
from sklearn.model_selection import train_test_split

SPACE = '<SPACE>'


def pad(D, max_syllables):
    """
    Takes the list of sublists in D and normalizes
    their length through cutting and zero-padding.
    Used to make the shape of the prescanned lines
    conform with the output of the syllable vectorizer.
    Returns a numpy array of shape: num instances x max_syllables.
    """
    X, x = [], []
    for line in D:
        x = line[:max_syllables]
        while len(x) < max_syllables:
            x.append(0.0)
        X.append(x)
    return np.array(X, dtype=np.float32)

def make_splits(syllables, stresses,
                random_state=12345, train_prop=0.9):
    """
    Wrapper to create a standard train/dev/test split,
    where the number of dev and test instances are both
    roughly equal to (1.0 - `train_prop`) / 0.5.
    No stratification is applied.
    """
    train_lines, rest_lines, train_stresses, rest_stresses = train_test_split(syllables, stresses,
                                                    train_size=train_prop, random_state=random_state)
    dev_lines, test_lines, dev_stresses, test_stresses = train_test_split(rest_lines, rest_stresses,
                                                 train_size=.5, random_state=random_state)

    return (train_lines, dev_lines, test_lines,
           train_stresses, dev_stresses, test_stresses)

class VerseIterator(object):

    def __init__(self, songsfile, max_songs=None, include_space=True,
                 return_type='syllables'):
        with open(songsfile, 'r') as f:
            self.songs = ujson.load(f)
        self.max_songs = max_songs
        self.include_space = include_space
        assert return_type in {'syllables', 'stresses'}
        self.return_type = return_type

    def __iter__(self):
        cnt = 0

        for song in self.songs:
            for verse in song['text']:
                for line in verse:
                    if self.return_type == 'syllables':
                        syllables = []
                        for word in line:
                            syllables.extend([w.lower() for w in word['syllables']])
                            if self.include_space:
                                syllables.append(SPACE)
                        yield syllables
                    elif self.return_type == 'stresses':
                        stresses = []
                        for word in line:
                            beats = [s for s in word['beatstress']]
                            beats = [int(s) if s in ('0', '1') else 0 for s in beats]
                            stresses.extend(beats)
                            if self.include_space:
                                stresses.append(0.0)
                        yield stresses

            cnt += 1
            if self.max_songs and cnt >= self.max_songs:
                break
