import random
random.seed(7676)
import numpy as np
import pandas as pd

def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1) * s)))

#topic_repr = 'topics_repr.csv'
#id2topic = pd.read_csv(topic_repr)
#id2topic.set_index('id', inplace=True, drop=True)
#del id2topic[id2topic.columns[0]]  # bug in pd?


def extract_artist(song):
    artist = song['artist'].strip().lower()
    artist = artist.replace('artist: ', '')
    artist = artist.replace(' ', '_')
    return artist

def extract_topic(song):
    id_ = song['id']
    topic_scores = id2topic.loc[id_]
    sampled_topic = list(id2topic.columns)[weighted_pick(topic_scores)]
    #sampled_topic = np.random.choice(id2topic.columns, 1,
    #                                 p=topic_scores)[0]
    return sampled_topic