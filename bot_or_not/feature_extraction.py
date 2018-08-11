import argparse
import json

from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial.distance as distance
import tqdm

from sklearn.feature_extraction.text import CountVectorizer


def entropy(items, base=None) -> float:
    _, counts = np.unique(filter(None, items), return_counts=True)
    return scipy.stats.entropy(counts)


def assonance_entropy(lines: List[str], phon_dict: Dict[str, str]) -> float:
    def stressed_nucleus(word):
        phones = phon_dict.get(word, [])
        return next((p for p in phones.split()[::-1] if p[-1] == "1"), None)

    scores = [entropy(stressed_nucleus(w) for w in line) for line in lines]
    return sum(scores) / len(scores)


def word_entropy(lines: List[str]) -> float:
    return entropy(word for line in lines for word in line)


def min_max_distance_to_corpus(self, items, samples) -> Tuple[float, float]:
    counts = self.vectorizer.transform([items])
    dists = [distance.cosine(counts, sample) for sample in samples]
    return min(dists), max(dists)


def group_syllables(line) -> List[List[str]]:
    words = []
    for syllable in line:
        if syllable.startswith("-"):
            words[-1].append(syllable)
        else:
            words.append([syllable])
    return words


def format_word(syllables: List[str]) -> str:
    if len(syllables) == 1:
        return syllables[0]

    def strip_hyphens(syllable):
        return syllable[syllable.startswith("-"):-syllable.endswith("-")]

    return "".join(strip_hyphens(syllable) for syllable in syllables)


def read_sample(sample: Dict, words=False) -> List[str]:
    lines = [line["original"].split() for line in sample["text"]]
    if words:
        lines = [[format_word(sylls) for sylls in group_syllables(line)]
                 for line in lines]
    return lines


class FeatureExtractor:
    def __init__(self, phon_path: str, min_freq: int = 5):
        self.min_freq = min_freq
        self.phon_dict = json.load(phon_path)

    def fit(self, org_samples, gen_samples) -> FeatureExtractor:
        org_samples = [
            read_sample(sample, words=True) for sample in org_samples
        ]
        gen_samples = [
            read_sample(sample, words=True) for sample in gen_samples
        ]

        self.vectorizer = CountVectorizer(analyzer=lambda t: t)
        self.vectorizer.fit(org_samples + gen_samples)

        self.X_org = self.vectorizer.transform(org_samples)
        self.X_gen = self.vectorizer.transform(gen_samples)

        feature_names = np.array(self.vectorizer.get_feature_names())
        self.low_frequency_words = set(
            feature_names[self.X_org.sum(axis=1) > self.min_freq])
        self.feature_names = set(feature_names)
        return self

    def transform(self, samples) -> List[Dict]:
        samples_features = []
        for sample in tqdm.tqdm(samples):
            features = {"sample_id": sample["id"]}
            syllable_lines = read_sample(sample)
            token_lines = read_sample(sample, words=True)
            # are there same word rhymes?
            features["same_word_rhyme"] = any(
                a[-1] == b[-1] for a, b in zip(lines, lines[1:]))
            # how many low-frequency words are there?
            features["low_frequency_word"] = sum(
                w in self.low_frequency_words for line in lines for w in line)
            # how many new words are there?
            features["new_word_frequency"] = sum(
                w not in self.feature_names for line in lines for w in line)
            # what's the average word length?
            features["avg_word_length"] = np.mean(
                [len(w) for line in lines for w in line])
            # what's the average syllable count per word?
            features["avg_syllable_count"] = np.mean([
                len(syllables) for line in syllable_lines
                for syllables in group_syllables(line)
            ])
            # how repetitive in terms of words is the same?
            features["repetitive_score"] = word_entropy(lines)
            # how coherent is the sample in terms of vowels?
            features["assonance_score"] = assonance_entropy(
                lines, self.phon_dict)
            # min and max distance to generated samples
            min_dist, max_dist = self.min_max_distance_to_corpus(
                lines, self.X_gen)
            features["min_dist_to_gen"] = min_dist
            features["max_dist_to_gen"] = max_dist
            # min and max distance to original samples
            min_dist, max_dist = self.min_max_distance_to_corpus(
                lines, self.X_org)
            features["min_dist_to_org"] = min_dist
            features["max_dist_to_gen"] = max_dist
            samples_features.append(features)
        return features

    def fit_transform(self, org_samples, gen_samples):
        self.fit(org_samples, gen_samples)
        return self.transform(org_samples), self.transform(gen_samples)


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

    extractor = FeatureExtractor(args.phon_path, args.min_freq)
    org_features, gen_features = extractor.fit_transform(
        org_samples, gen_samples)
    features = pd.DataFrame(org_samples + gen_samples)
