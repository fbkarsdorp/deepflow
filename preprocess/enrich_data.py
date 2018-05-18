import json

with open("mcflow.json") as f:
    data = json.load(f)

mcflow_syllables = {}
for song in data:
    for verse in song['text']:
        for line in verse:
            for word in line:
                mcflow_syllables[word['word'].lower()] = word['syllables']

syllables = [list(map(int, line.strip().split())) for line in open('mcflow.vocab.tagged.txt')]
vocabulary = [line.strip() for line in open('mcflow.vocab.txt')]

assert len(syllables) == len(vocabulary)

stress_dict = {}
stress_dict['man'] = [1]
for i, (word, sylls) in enumerate(zip(vocabulary, syllables), 1):
    n_syllables = sum(elt > 0 for elt in sylls)
    if len(mcflow_syllables[word]) == 1:
        sylls = [2] + [0] * (len(word) - 1)
        stress_dict[word] = [1 if elt == 2 else 0 for elt in sylls if elt > 0]
    elif n_syllables != len(mcflow_syllables[word]):
        print(i, word, sylls, mcflow_syllables[word])
    else:
        stress_dict[word] = [1 if elt == 2 else 0 for elt in sylls if elt > 0]

for song in data:
    for verse in song['text']:
        for line in verse:
            for word in line:
                if len(word['syllables']) != len(stress_dict[word['word'].lower()]):
                    stress = input(f'{word["word"]}, {word["syllables"]} ').strip()
                    stress = list(map(int, stress.split()))
                    if len(stress) != len(word['syllables']):
                        stress = input(f'{word["word"]}, {word["syllables"]} ').strip()
                        stress = list(map(int, stress.split()))                    
                else:
                    stress = stress_dict[word['word'].lower()]
                word['stress'] = stress
    
