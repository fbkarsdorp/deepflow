import collections
import json
import os
import pickle

from prompt_toolkit import prompt, auto_suggest

with open("../data/mcflow/mcflow.json") as f:
    data = json.load(f)


class Suggestion(auto_suggest.AutoSuggest):
    def __init__(self, suggestion):
        self.suggestion = auto_suggest.Suggestion(suggestion)
        
    def get_suggestion(self, cli, buffer, document):
        return self.suggestion

mcflow_syllables = {}
for song in data:
    for verse in song['text']:
        for line in verse:
            for word in line:
                mcflow_syllables[word['word'].lower()] = word['syllables']

syllables = [list(map(int, line.strip().split())) for line in open('../data/lexica/mcflow.vocab.primary.tagged.txt')]
vocabulary = [line.strip() for line in open('../data/lexica/mcflow.vocab.txt')]

assert len(syllables) == len(vocabulary)

if os.path.exists('../data/lexica/mcflow.vocab.primary.pickle'):
    with open('../data/lexica/mcflow.vocab.primary.pickle', 'rb') as f:
        curated_stress_dict = pickle.load(f)
    stress_dict = {}
    for key, stress in curated_stress_dict:
        stress_dict[''.join(key)] = stress
else:
    stress_dict = {}
    for i, (word, sylls) in enumerate(zip(vocabulary, syllables), 1):
        n_syllables = sum(elt > 0 for elt in sylls)
        if len(mcflow_syllables[word]) == 1:
            sylls = [2] + [0] * (len(word) - 1)
            stress_dict[word] = [1 if elt == 2 else 0 for elt in sylls if elt > 0]
        elif n_syllables != len(mcflow_syllables[word]):
            annotation = prompt(f'Correct: {mcflow_syllables[word]} ({" ".join(map(str, sylls))}): ')
            stress_dict[word] = list(map(int, annotation.strip().split()))
        else:
            stress_dict[word] = [1 if elt == 2 else 0 for elt in sylls if elt > 0]
    curated_stress_dict = {}


for song in data:
    for verse in song['text']:
        for line in verse:
            for word in line:
                if len(word['syllables']) != len(stress_dict[word['word'].lower()]):
                    if curated_stress_dict.get(tuple(word['syllables'])) is not None:
                        stress = curated_stress_dict.get(tuple(word['syllables']))
                    else:
                        stress = []
                        while len(stress) != len(word['syllables']):
                            stress = prompt(f'{word["word"]}, {word["syllables"]} ')
                            stress = [int(elt) for elt in stress.strip().split()]
                        curated_stress_dict[tuple(word['syllables'])] = tuple(stress)
                else:
                    stress = stress_dict[word['word'].lower()]
                    curated_stress_dict[tuple(word['syllables'])] = tuple(stress_dict[word['word'].lower()])
                word['stress'] = stress
    
