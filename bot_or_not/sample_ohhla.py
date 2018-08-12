import argparse
import json
import random
import uuid

from typing import List, Dict


def hyphenate(syllables) -> List[str]:
    if len(syllables) == 1:
        return syllables
    output = []
    for i, syllable in enumerate(syllables):
        if i == 0:
            syllable = syllable + '-'
        elif i == (len(syllables) - 1):
            syllable = '-' + syllable
        else:
            syllable = '-' + syllable + '-'
        output.append(syllable)
    return output


def sample_from_ohhla(fpath: str, n_samples: int = 10000) -> List[Dict]:
    samples = []
    for line in open(fpath):
        song = json.loads(line.strip())
        for verse in song['text']:
            while len(verse) >= 2:

                if len(samples) == n_samples:
                    return samples
                
                idx = random.randint(2, min(len(verse), 4))
                lines, verse = verse[:idx], verse[idx:]
                sample = {
                    'id': str(uuid.uuid1())[:8],
                    'song_id': song['id'],
                    'artist': song['artist'],
                    'album': song['album'],
                    'text': [{
                        'original':
                        ' '.join(
                            sum((hyphenate(word['syllables'])
                                 if word['syllables'] else [word['token']]
                                 for word in line), []))
                    } for line in lines]
                }
                samples.append(sample)
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipath', type=str)
    parser.add_argument('--opath', type=str)
    parser.add_argument('--samples', type=int, default=10000)
    args = parser.parse_args()

    samples = sample_from_ohhla(args.ipath, n_samples = args.samples)
    with open(args.opath, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

