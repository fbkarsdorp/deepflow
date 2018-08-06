import collections
import json
import random

from typing import Dict, List

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

        # starting values
        self.level = 0
        self.iteration = 0
        self.seen = set()

        # set up examples in bins
        if fpath is not None:
            with open(fpath) as f:
                examples = json.load(f)
        else:
            examples = examples
        self.pairs = bin_examples(examples, bins=levels)

    def next(self) -> Dict:
        if self.iteration == self.n_iter and self.level != self.levels:
            self.level += 1
            self.iteration = 0
            print('Level UP!')
        # check if there are
        n_pairs = len(self.pairs[self.level])
        if n_pairs == 0:
            if self.level == self.levels:
                return "GAME OVER"
            self.level += 1
        # sample a new pair for the current level
        self.iteration += 1
        return self.pairs[self.level].pop(random.randint(0, n_pairs))
