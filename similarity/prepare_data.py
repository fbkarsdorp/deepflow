import argparse
import json
import re
from typing import List

from nltk import word_tokenize
from sklearn.model_selection import train_test_split

def format_example(example, real=True) -> str:
    return f'__label__{1 if real else 0} {example}'

def clean_example(example) -> str:
    return ' '.join(word_tokenize(example)).lower()

def sample_from_ohhla(fpath: str, n: int = 10000) -> List[str]:
    examples = []
    for line in open(fpath):
        song = json.loads(line.strip())
        verse = random.choice(song['text'])
        lines = verse[: random.randint(0, len(verse))]
        lines = [word['token'] for line in lines for word in line]
        examples.append(format_example(clean_example('\n'.join(lines))))
        if len(examples) == n:
            break
    return examples

def read_examples(fpath: str, real=True) -> List[str]:
    examples = []
    with open(fpath) as f:
        data = f.read()
    for example in re.finditer('\n\n', data):
        examples.append(format_example(clean_example(example), real=real))
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--real_input', type=str, help='file with real instances')
    parser.add_argument(
        '--fake_input', type=str, help='file with fake instances')
    parser.add_argument('--output_prefix', type=str, help='output file prefix')
    args = parser.parse_args()

    real_examples = sample_from_ohhla(args.real_input)
    fake_examples = read_examples(args.fake_input, real=False)

    train, dev = train_test_split(
        real_examples + fake_examples, test_size=0.1, shuffle=True)

    with open(f'{args.output_prefix}.train', 'w') as out:
        out.write('\n'.join(train))

    with open(f'{args.output_prefix}.dev', 'w') as out:
        out.write('\n'.join(dev))
