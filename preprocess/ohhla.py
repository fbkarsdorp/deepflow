import argparse
import csv
import os
import re

import bs4 as bs
import pandas as pd


excluded_files = 'politics.r59.txt', '00readme.txt', 'readme.txt', 'robots.txt'

artist_variations = (
    'Arist', 'Author', 'Artitst', 'Artist', 'Artsist', 'rtist',
    'Aritst', 'Artst', 'Artis', 'Atrist', 'Artsit', 'Arstist',
    'FArtist', 'Rrtist', 'Artest', 'Artit', 'Artists'
)

album_variations = (
    'album:', 'abum:', 'ablum:', 'albun:', 'albu:', 'alubm:', 'alvum:', 'albym:', 'albim:', 'albums:'
)

song_variations = (
    'song:', 'title:', 'soung:', 'spng:', 'solg:', 'sont:', 'track:', 'sonh:', 'somg:', 'songs:'
)


def iter_corpus(path):
    done = set()
    for root, dirs, files in os.walk(path):
        if root.split('/')[-1].startswith('?'):
            continue
        for f in files:
            if f'{root}/{f}' in done:
                continue
            if f.endswith('.txt') and f not in excluded_files:
                  with open(f'{root}/{f}', encoding='latin-1') as infile:
                      tree = bs.BeautifulSoup(infile.read())
                      pre = tree.find('pre')
                      if pre is None:
                          text = tree.find('body').get_text().strip()
                          assert text.startswith(artist_variations), tree
                      else:
                          text = pre.get_text().strip()
                          if text == '' or text.startswith(('Warning:  include', 'Parse error')):
                              continue
                          assert pre.get_text().strip().startswith(artist_variations), tree
                  done.add(f'{root}/{f}')
                  yield text

def collect_meta(text):
    meta = {}
    for i, line in enumerate(text):
        if line.startswith(artist_variations):
            if 'artist' in meta:
                meta['album'] = line
            else:
                meta['artist'] = line
        elif line.lower().startswith(album_variations):
            if 'album' in meta:
                meta['song'] = line
            else:
                meta['album'] = line
        elif line.lower().startswith(song_variations):
            meta['song'] = line
        elif not line.strip() and len(meta) == 3:
            break
    return meta, i + 1

numbers = '|'.join('one two three four five six seven eight nine ten'.split())
clean_re = re.compile(fr'''
    (^\[.*?\]:?$)                                 # remove all comments between square brackets
 |  (^\[.*?\]\s(\+\s)?[({{].*?[)}}]$)             # square brackets plus voice indicator
 |  (^[({{]?[Vv]erse\s(\d+|{numbers}:))            # Remove verse start indicators
 |  (^\d((st)|(nd)|(d))\s Verse:?$)               # more verses
 |  (^[{{(]?\*.*?\*[}})]?$)                       # Remove text between asterisks
 |  (^[{{[(-~*]?[cC]horus[:*]?)                   # Remove chorus indicators
 |  (^\(+[Cc]horus\)+$)                           # Remove chorus indication in parentheses
 |  (^Repeat\sChorus)                             # Corpus repetitions
 |  (^(->\s)?\(?Repeat\s((\d+X?)|(x\s?\d+))\)?$)  # repetitions
 |  (^\(?((x\d+)|(\d+x))\)?$)                     # short repetitions
 |  (^\(Hook(\s(\d+\s?X?)|(x\s\d+))?\))           # hooks
 |  (^Hook\s((\d+\s?x)|(x\s?\d+))$)
 |  (^\[Hook\]\s(-\s)?((\d+\s?x)|(x\s?\d+))$)
 |  (^\(Hook[:,]\s.*?\)$)
 |  (^Hook(\srepeat)?:?$)
 |  (^Repeat(\shook)?$)
 |  (^\[?Intro(:|$))
 |  (^[[(]?Outro\)?(:|$))
 |  (^Intro/((Outro)|(Chorus)))
 |  (^Break(\s\d+)?:?$)                           # break it
 |  (^Bridge(\s|:|$))                             # take me to the bridge
 |  (^\*\ssend\scorrections\sto\sthe\stypist)     # corrections
 |  (^\*\shelp\srequested)                        # help
 |  (^\([\w.-]+\)$)                               # mostly names of singers (precision could be better)
''', re.VERBOSE | re.IGNORECASE)

singer_re = re.compile(r'^((\[.*?\])|(\(([A-Z]([a-z.]+)?){1,2}\)))\s?')
repetions_re = re.compile(r'[{(]\d+\s?[Xx][})]\s?')

def parse_song(text):
    text = text.splitlines()
    header, song_start = collect_meta(text)
    lines = []
    lines = [repetions_re.sub('', singer_re.sub('', singer_re.sub('', line.strip())))
             for line in text[song_start:] if clean_re.match(line.strip()) is None]
    verses = [verse.split('\n') for verse in re.split(r'\n{2,}', '\n'.join(lines))]
    verses = '\n\n'.join('\n'.join(verse) for verse in verses if len(verse) > 1)
    header['text'] = verses
    return header

if __name__ == '__main__':
    with open("../data/ohhla.csv", "w") as out:
        writer = csv.DictWriter(out, ['artist', 'album', 'song', 'text'])
        writer.writeheader()
        for text in iter_corpus('../data/ohhla.com'):
            song = parse_song(text)
            writer.writerow(song)

        
