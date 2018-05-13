import csv
import spacy
import ujson
import tqdm


TAGGER = spacy.load('en', disable=['parser', 'ner'])

with open("../data/ohhla.csv") as f:
    reader = csv.DictReader(f, ['artist', 'album', 'song', 'text'])
    next(reader) # skip header
    data = list(reader)

songs = []
for i, entry in tqdm.tqdm(enumerate(data), total=len(data)):
    text = entry['text']
    json = {
        'id': i, 'artist': entry['artist'], 'album': entry['album'],
        'song': entry['song'], 'text': []
    }
    verses = []
    json_repr = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            json_repr.append([{'token': word.orth_, 'pos': word.pos_} for word in TAGGER(line)])
        else:
            verses.append(json_repr)
            json_repr = []
    if json_repr:
        verses[-1].append(json_repr)
    json['text'] = verses
    songs.append(json)

with open('../data/ohhla.json', 'w') as outfile:
    ujson.dump(songs, outfile, ensure_ascii=False, escape_forward_slashes=False)
