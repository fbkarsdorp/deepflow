
import ijson


def get_syllables(syllables):
    for idx, syl in enumerate(syllables):
        if idx == 0:
            if len(syllables) > 1:
                yield syl + '-'
            else:
                yield syl
        elif idx == len(syllables) - 1:
            yield '-' + syl
        else:
            yield '-' + syl + '-'
        

def get_text(fpath, syllable_level=False):
    with open(fpath) as f:
        for song in ijson.items(f, 'item'):
            for verse in song['text']:
                for line in verse:
                    if syllable_level:
                        output = ''
                        for item in line:
                            output += ' '.join(get_syllables(item['syllables']))
                            output += ' '
                        yield output
                    else:
                        yield ' '.join(elt['token'] for elt in line)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--syllable_level', action='store_true')
    args = parser.parse_args()
    with open('./data/ohhla-beatstress{}.txt'.format('.syll' if args.syllable_level else ''), 'w+') as f:
        for line in get_text('./data/ohhla-beatstress.json', syllable_level=args.syllable_level):
            f.write(line + '\n')
