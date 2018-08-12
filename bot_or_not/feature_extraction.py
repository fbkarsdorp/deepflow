import argparse
import itertools
import json

from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics.pairwise import pairwise_distances
import tqdm

from sklearn.feature_extraction.text import CountVectorizer


VOWELS = 'AA AE AH AO AW AX AY EH ER EY IH IX IY OW OY UH UW UX'.split()

def entropy(items, base=None) -> float:
    _, counts = np.lib.arraysetops.unique(list(filter(None, items)), return_counts=True)
    return scipy.stats.entropy(counts)


def assonance_entropy(lines: List[str], phon_dict: Dict[str, str]) -> float:
    def stressed_nucleus(word):
        phones = phon_dict.get(word, '')
        return next((p for p in phones.split()[::-1] if p[-1] == "1"), None)

    scores = [entropy(stressed_nucleus(w) for w in line) for line in lines]
    return sum(scores) / len(scores)


def onset(word, phon_dict):
    phones = phon_dict.get(word, None)
    strip_stress = lambda p: p[:-1] if p.endswith(('1', '2')) else p
    if phones is not None:
        phones = ''.join(itertools.takewhile(
            lambda p: strip_stress(p) not in VOWELS, phones.split()))
    return phones


def onset_entropy(lines: List[str], phon_dict: Dict[str, str]) -> float:
    scores = [entropy(onset(w, phon_dict) for w in line) for line in lines]
    return sum(scores) / len(scores)


def alliteration_score(lines: List[str], phon_dict: Dict[str, str]) -> int:
    alliterations = 0
    for line in lines:
        for a, b in zip(line, line[1:]):
            onset_a, onset_b = onset(a, phon_dict), onset(b, phon_dict)
            if onset_a is not None and onset_a == onset_b:
                alliterations += 1
    return alliterations / len(sum(lines, []))
    

def vocab_entropy(lines: List[str]) -> float:
    return entropy(word for line in lines for word in line)


def group_syllables(line) -> List[List[str]]:
    words = []
    for syllable in line:
        if syllable.startswith("-"):
            if not words:
                words.append([])
            words[-1].append(syllable)
        else:
            words.append([syllable])
    return words


def format_word(syllables: List[str]) -> str:
    if len(syllables) == 1:
        return syllables[0]

    def strip_hyphens(syllable):
        return syllable[syllable.startswith("-"):-syllable.endswith("-") or None]

    return "".join(strip_hyphens(syllable) for syllable in syllables)


def read_sample(sample: Dict, words=False) -> List[str]:
    lines = [line["original"].split() for line in sample["text"]]
    if words:
        lines = [[format_word(sylls) for sylls in group_syllables(line)]
                 for line in lines]
    return lines


class FeatureExtractor:
    def __init__(self, phon_path: str, min_freq: int = 5, debug: bool = True):
        self.min_freq = min_freq
        self.debug = debug
        with open(phon_path) as f:
            self.phon_dict = json.load(f)

    def fit(self, org_samples, gen_samples):
        org_samples = [
            read_sample(sample, words=True) for sample in tqdm.tqdm(org_samples)
        ]
        gen_samples = [
            read_sample(sample, words=True) for sample in tqdm.tqdm(gen_samples)
        ]
        self.vectorizer = CountVectorizer(analyzer=lambda t: sum(t, []))
        self.vectorizer.fit(org_samples + gen_samples)

        self.X_org = self.vectorizer.transform(org_samples)
        self.X_gen = self.vectorizer.transform(gen_samples)

        feature_names = np.array(self.vectorizer.get_feature_names())
        self.low_frequency_words = set(
            feature_names[(self.X_org.sum(axis=0) <= self.min_freq).A[0]])
        self.feature_names = set(feature_names)
        return self

    def transform(self, samples, original) -> List[Dict]:
        samples_features = []
        for idx, sample in tqdm.tqdm(enumerate(samples)):
            features = {"sample_id": sample["id"]}
            syllable_lines = read_sample(sample)
            token_lines = read_sample(sample, words=True)
            # are there same word rhymes?
            features["same_word_rhyme"] = any(
                a[-1] == b[-1] for a, b in zip(token_lines, token_lines[1:]))
            # how many low-frequency words are there?
            features["low_frequency_word"] = sum(
                w in self.low_frequency_words for line in token_lines for w in line)
            # how many new words are there?
            features["new_word_frequency"] = sum(
                w not in self.feature_names for line in token_lines for w in line)
            # what's the average word length?
            features["avg_word_length"] = np.mean(
                [len(w) for line in token_lines for w in line])
            # what's the average syllable count per word?
            features["avg_syllable_count"] = np.mean([
                len(syllables) for line in syllable_lines
                for syllables in group_syllables(line)
            ])
            # how repetitive in terms of words is the fragment?
            features["word_repetitive_score"] = vocab_entropy(token_lines)
            # how repetitive in terms of syllables is the fragment?
            features["syllable_repetitive_score"] = vocab_entropy(syllable_lines)
            # how coherent is the sample in terms of vowels?
            features["assonance_score"] = assonance_entropy(
                token_lines, self.phon_dict)
            # how coherent is the sample in terms of onsets?
            features["onset_score"] = onset_entropy(
                token_lines, self.phon_dict)
            features['alliteration_score'] = alliteration_score(
                token_lines, self.phon_dict)
            # min and max distance to generated samples
            dist_scores = self.distances_to_samples(idx, original)
            features.update(dist_scores)
            features['label'] = int(original)
            samples_features.append(features)
        return samples_features    

    def fit_transform(self, org_samples, gen_samples) -> Tuple[Dict, Dict]:
        self.fit(org_samples, gen_samples)
        return self.transform(org_samples, True), self.transform(gen_samples, False)

    def distances_to_samples(self, idx, original) -> Dict[str, float]:
        counts = self.X_org[idx] if original else self.X_gen[idx]
        dists2org = pairwise_distances(counts, self.X_org, metric='cosine')[0]
        dists2org.sort()
        dists2gen = pairwise_distances(counts, self.X_gen, metric='cosine')[0]
        dists2gen.sort()
        scores = {}
        if original:
            scores['min_dist2org'] = dists2org[1]
            scores['max_dist2org'] = dists2org[-1]
            scores['min_dist2gen'] = dists2gen[0]
            scores['max_dist2gen'] = dists2gen[-1]
        else:
            scores['min_dist2org'] = dists2org[0]
            scores['max_dist2org'] = dists2org[-1]
            scores['min_dist2gen'] = dists2gen[1]
            scores['max_dist2gen'] = dists2gen[-1]            
        return scores
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--org_sample_path', type=str)
    parser.add_argument('--gen_sample_path', type=str)
    parser.add_argument('--phon_dict_path', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--min_freq', type=int)
    args = parser.parse_args()

    org_samples = [
        json.loads(line.strip()) for line in open(args.org_sample_path)
    ]
    gen_samples = [
        json.loads(line.strip()) for line in open(args.gen_sample_path)
    ]

    extractor = FeatureExtractor(args.phon_dict_path, args.min_freq)
    org_features, gen_features = extractor.fit_transform(
        org_samples, gen_samples)
    features = pd.DataFrame(org_features + gen_features)
