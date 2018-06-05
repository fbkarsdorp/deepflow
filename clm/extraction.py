import random
random.seed(7676)
import numpy as np
import pandas as pd

from sklearn.externals import joblib


def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1) * s)))


class Extractor(object):

    def __init__(self, conditions, topics_file='topics_repr.csv',
                 topic_vectorizer='topics_vector.pkl',
                 orig_nmf='topics_nmf.pkl'):
        self.getters = {'artists': self.extract_artist,
                        'topics': self.extract_topic}
        self.id2topic = pd.read_csv(topics_file)
        self.id2topic.set_index('id', inplace=True, drop=True)
        del self.id2topic[self.id2topic.columns[0]]  # bug in pandas?

        nmf = joblib.load(orig_nmf)
        vectorizer = joblib.load(topic_vectorizer)

        for topic_idx, topic in enumerate(nmf.components_):
            top_idxs = np.argsort(topic)[::-1][:25]
            top_words = np.array(vectorizer.get_feature_names())[top_idxs]
            print('topic' + str(topic_idx) + ' -> ' + ' - '.join(top_words))

    def __getitem__(self, key):
        return self.getters[key]

    def extract_artist(self, song):
        artist = song['artist'].strip().lower()
        artist = artist.replace('artist: ', '')
        artist = artist.replace(' ', '_')
        return artist

    def extract_stress(self, song):
        for verse in song['text']:
            for line in verse:
                syllables.append('<BR>')
                for word in line:
                    for i, s in enumerate(word['beatstress']):
                        items.append(s)
        return syllables

    def extract_topic(self, song):
        id_ = song['id']
        topic_scores = self.id2topic.loc[id_]
        sampled_topic = list(self.id2topic.columns)[weighted_pick(topic_scores)]
        #sampled_topic = np.random.choice(id2topic.columns, 1,
        #                                 p=topic_scores)[0]
        return sampled_topic
