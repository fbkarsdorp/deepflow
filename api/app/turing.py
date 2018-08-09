import collections
import json
import random

from typing import Dict, List, Tuple

import pandas as pd


def bin_examples(examples: List[Dict], bins=10) -> Dict[int, Dict]:
    scores = [example['score'] for example in examples]
    labels = pd.cut(scores, bins, labels=False)
    binned_examples = collections.defaultdict(list)
    for example, label in zip(examples, labels):
        binned_examples[label].append(example)
    return binned_examples


class ExampleSampler:
    def __init__(self, fpath: str, levels: int = 10, n_iter: int = 5):
        self.levels = levels
        self.n_iter = n_iter

        # set up examples in bins
        with open(fpath) as f:
            examples = json.load(f)
        self.pairs = bin_examples(examples, bins=levels)

    def next(self, iteration: int, level: int, seen: List[int]) -> Tuple:
        seen = set(seen)
        if iteration == self.n_iter and level != self.levels:
            level += 1
            iteration = 0
        if not self.pairs[level]:
            if level == self.levels:
                return {'id': None, 'false': None, 'true': None}
            level += 1
            iteration = 0
        # sample a new pair for the current level
        id, true, false = None, None, None
        while id is None:
            candidate = random.choice(self.pairs[level])
            if candidate['id'] not in seen:
                id, true, false = candidate['id'], candidate['true'], candidate['false']
                seen.add(id)
        return id, true, false, iteration + 1, level
