import random
import json
import re
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

splitter = re.compile(r'\s+f\/|, |\(|& | a\/k\/a |w\/ |feat\/| f\.\/ |a\.k\.a\.|  aka |\/|\+\s+| \-\- | \- |\[|\{|\s+\~\s+')

def stringify(artists, unicize=True):
    if unicize:
        artists = list(set(artists))
    random.shuffle(artists)

    data = ['#']
    for artist in artists:
        data.extend([a for a in artist if a])
        data += ['#']

    return tuple(data)


def load_artists(fn, max_artists=None):
    artists = []

    for song_idx, song in enumerate(open(fn)):
        song = json.loads(song)

        # cleanup:
        artist_info = song['artist']
        artist_info = re.sub(r'Artist:|Arstist:|Artsist:|', '', artist_info)
        artist_info = re.sub(r'\s+', ' ', artist_info)
        artist_info = artist_info.replace('"', '').replace('#', '')
        artist_info = ''.join([c for c in artist_info if c.isprintable()])

        # split collaborations:
        artists_ = splitter.split(artist_info)
        artists_ = [a.replace('*', '').replace(')', '').replace(']', '').replace('}', '') for a in artists_]
        artists_ = [a.replace("'$", '').replace("^'", '').replace('"$', '').replace('^"', '') for a in artists_]
        artists_ = [a.strip() for a in artists_ if a.strip()]
        artists.extend(artists_)

        if max_artists and song_idx >= max_artists:
            break

    return artists


def char_tensor(sequence, vocabulary):
    tensor = torch.zeros(len(sequence)).long()
    for c in range(len(sequence)):
        tensor[c] = vocabulary.index(sequence[c])
    return Variable(tensor)


def random_batch(data, vocabulary, bptt):
    start_index = random.randint(0, len(data) - bptt)
    end_index = start_index + bptt + 1
    chunk = data[start_index : end_index]
    
    inp = char_tensor(chunk[:-1], vocabulary)
    target = char_tensor(chunk[1:], vocabulary)
    return inp, target


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def generate(model, vocabulary, prime_str='#', predict_len=100, temperature=0.8):
    hidden = model.init_hidden()
    prime_input = char_tensor(prime_str, vocabulary)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = model(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = vocabulary[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char, vocabulary)

    return predicted