import csv
import spacy
import ujson


TAGGER = spacy.load('en', disable=['parser', 'ner'])

with open("../data/ohhla.csv") as f:
    reader = csv.DictReader(f, ['artist', 'album', 'song', 'text'])
    next(reader) # skip header

songs = []
for i, entry in enumerate(data):
    text = entry['text']
    json = {
        'id': i, 'artist': entry['artist'], 'album': entry['album'],
        'song': entry['song'], 'text': []
    }
    for line in text.splitlines():
        line = line.strip()
        if line:
            json_repr = []
            tagged_line = TAGGER(line)
            for word in tagged_line:
                json_repr.append({'token': word.orth_, 'pos': word.pos_})
            json['text'].append(json_repr)
    songs.append(json)

with open('../data/ohhla.json', 'w') as outfile:
    ujson.dump(songs, outfile, ensure_ascii=False, escape_forward_slashes=False)
