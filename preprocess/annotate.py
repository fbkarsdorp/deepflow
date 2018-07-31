import csv
import json
import tqdm
import ucto

tokenizer = ucto.Tokenizer("tokconfig-eng")

contractions = {
    "ain't", "aren't", "can't", "could've", "couldn't", "daren't", "daresn't", "dasn't", "didn't", 
    "doesn't", "don't", "e'er", "everyone's", "finna", "gimme", "gonna", "gotta", "hadn't", "hasn't",
    "haven't", "he'd", "he'll", "he's", "he've", "how'd", "how'll", "how're", "how's", "I'd", "I'll", 
    "I'm", "I'm'a", "I'm'o", "I've", "isn't", "it'd", "it'll", "it's", "let's", "ma'am", "mayn't", "may've", 
    "mightn't", "might've", "mustn't", "mustn't've", "must've", "needn't", "ne'er", "o'clock", "o'er", 
    "ol'", "oughtn't", "shan't", "she'd", "she'll", "she's", "should've", "shouldn't", "somebody's", 
    "someone's", "something's", "that'll", "that're", "that's", "that'd", "there'd", "there'll", "there're", 
    "there's", "these're", "they'd", "they'll", "they're", "they've", "this's", "those're", "'tis", 
    "'twas", "wasn't", "we'd", "we'd've", "we'll", "we're", "we've", "weren't", "what'd", "what'll", 
    "what've", "when's", "where'd", "where're", "where's", "where've", "which's", "who'd", "who'd've", 
    "who'll", "who're", "who's", "who've", "why'd", "why're", "why's", "won't", "would've", "wouldn't", 
    "y'all", "y'all'd've", "yesn't", "you'd", "you'll", "you're", "you've", "noun's"}

def resolve_contractions(tokens):
    new_tokens = []
    for i, token in enumerate(tokens):
        if new_tokens and (new_tokens[-1] + token) in contractions:
            new_tokens[-1] += token
        elif new_tokens and token in ("'ll", "'m", "'t", "'d", "'ve", "'re", "'s", "'am"):
            new_tokens[-1] += token
        elif new_tokens and token == "n't":
            new_tokens[-1] += token
        elif new_tokens and token == "'" and new_tokens[-1].endswith('in'):
            new_tokens[-1] += token
        else:
            new_tokens.append(token)
    return new_tokens

with open("../data/lyrics-corpora/ohhla.csv") as f:
    reader = csv.DictReader(f, ['artist', 'album', 'song', 'text'])
    next(reader) # skip header
    data = list(reader)

outfile = open('../data/lyrics-corpora/ohhla-corrected.json', 'w')
for i, entry in tqdm.tqdm(enumerate(data), total=len(data)):
    text = entry['text']
    json_dict = {
        'id': i, 'artist': entry['artist'], 'album': entry['album'],
        'song': entry['song'], 'text': []
    }
    verses = []
    json_repr = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
            tokenizer.process(line)
            tokens = [word for sentence in tokenizer.sentences() for word in sentence.split()]
            tokens = [{'token': token} for token in resolve_contractions(tokens)]
            json_repr.append(tokens)
        else:
            verses.append(json_repr)
            json_repr = []
    if json_repr:
        verses.append(json_repr)
    json_dict['text'] = verses
    outfile.write(json.dumps(json_dict) + '\n')


