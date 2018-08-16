import argparse
import collections
import json
import random
import re
import sys
import uuid

sys.path.append('../api/app')
from generation.utils import detokenize, join_syllables

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


ARTIST_PREFIX_RE = re.compile(r'[Aa]rtis?t?:?\s+')
ALBUM_PREFIX_RE = re.compile(r'[Aa]lbum?:?\s+')

def clean_field(field, regex):
    match = regex.match(field)
    if match is None:
        return field.strip()
    return field[match.end():].strip()


parser = argparse.ArgumentParser()
parser.add_argument('--score_file', type=str)
parser.add_argument('--maximum_score', type=float, default=0.8)
parser.add_argument('--sample_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--bins', type=int, default=10)
args = parser.parse_args()

df = pd.read_csv(args.score_file)
columns = [c for c in df.columns if c not in ('sample_id', 'label')]

X, y = df[columns].values, df['label'].values

lr = LogisticRegression().fit(X, y)
probs = lr.predict_proba(X)

average_precision = average_precision_score(y, probs[:,1])
print(f'AP: {average_precision:.2f}')

# compute insecurity of classifier
df['score'] = abs(probs[:,0] - probs[:,1])
df = df[df['score'] <= args.maximum_score]

# scale the scores between 0 and 1
df['score'] = 1 - ((df['score'] - df['score'].min()) /
                   (df['score'].max() - df['score'].min()))

samples = {}
for line in open(args.sample_file):
    sample = json.loads(line.strip())
    samples[sample['id']] = sample

for i, row in df.iterrows():
    samples[row['sample_id']]['score'] = row['score']
    samples[row['sample_id']]['label'] = row['label']

samples = [sample for sample in samples.values() if 'score' in sample]
scores = [sample['score'] for sample in samples]
labels = pd.cut(scores, bins=args.bins, labels=False)
binned_samples = collections.defaultdict(list)
for sample, label in zip(samples, labels):
    binned_samples[label].append(sample)

taken = set()
pairs = []
for samples in binned_samples.values():
    print(len(samples))
    n_pairs = 0
    original_samples = [sample for sample in samples if sample['label'] == 1]
    random.shuffle(original_samples)
    generated_samples = [sample for sample in samples if sample['label'] == 0]
    random.shuffle(generated_samples)
    for original, generated in zip(original_samples, generated_samples):
        pair = {
            'id': str(uuid.uuid1())[:8],
            'score': (original['score'] + generated['score']) / 2,
            'true_id': original['id'],
            'false_id': generated['id'],
            'true': [detokenize(join_syllables(line['original'].split()))
                     for line in original['text']],
            'artist': clean_field(original['artist'], ARTIST_PREFIX_RE),
            'album': clean_field(original['album'], ALBUM_PREFIX_RE),
            'false': [detokenize(join_syllables(line['original'].split()))
                      for line in generated['text']]
        }
        pairs.append(pair)
        n_pairs += 1
    print(f'Formed {n_pairs} pairs')

with open(args.output_file, 'w') as f:
    for pair in pairs:
        f.write(json.dumps(pair) + '\n')
            
