import os
import re
import sys
import json

def parse_celex(filepath, column_idx=None):
    lexicon = {}
    for line in open(filepath):
        fields = line.strip().split('\\')
        if fields[1] in lexicon:
            print(fields[1])
        lexicon[fields[1]] = fields[column_idx]
    return lexicon

data = {}
phono_lexicon = parse_celex(sys.argv[1], column_idx=4)
syllable_lexicon = parse_celex(sys.argv[2], column_idx=8)
lexicon = []
data = {}
errors, correct = 0, 0
for word, phon_str in phono_lexicon.items():
    if word not in syllable_lexicon:
        errors += 1
        continue
    elif ' ' in word:
        continue
    syllables = syllable_lexicon[word].replace('--', '$-')
    syllables = re.split('-+| ', syllables)
    syllables = [re.sub('\$$', '-', s) for s in syllables]
    phonology = phon_str.split('-')
    print(word, syllables, phonology, phon_str, syllable_lexicon[word])
    if len(syllables) != len(phonology):
        errors += 1
        continue # mostly errors...
    stress = [1 if s.startswith("'") else 2 if s.startswith('"') else 0 for s in phonology]
    assert stress
    data[word] = {'syllables': syllables, 'stress': stress}
    correct += 1

# for line in open(sys.argv[3]):
#     syllables = re.split('</[012]>', line.lower().strip())[:-1]
#     stress = [1 if s.startswith('<1>') else 2 if s.startswith('<2>') else 0 for s in syllables]
#     syllables = [s[3:] for s in syllables]
#     if ''.join(syllables) not in data:
#         print(''.join(syllables))
#         data[''.join(syllables)] = {'syllables': syllables, 'stress': stress}

with open(sys.argv[4], 'w') as out:
    json.dump(data, out, indent=4)

