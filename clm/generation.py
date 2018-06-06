import os
import glob
from itertools import product

from keras.models import load_model

from vectorization import SequenceVectorizer
from modelling import sample

def main():
    model_path = 'clm_model'
    vectorizer_dir = 'vectorizers'
    max_len = 25
    
    artists = ['eminem']
    topics = ['t40', 't7']
    rhythms = ['r6', 'r0']
    seeds = [['<BOS>', '<BR>'],
             ['i/', "'m", 'a', 'real']]
    temperatures = [0.1, 0.3, 0.5]

    model = load_model(model_path)
    
    vectorizers = {}
    paths = sorted(glob.glob(vectorizer_dir + '/*.json'))
    for n in paths:
        v = SequenceVectorizer.load(n)
        n = os.path.basename(n).replace('.json', '')
        vectorizers[n] = v

    bptt = vectorizers['syllables'].bptt

    for artist, topic, rhythm, seed, temp in product(artists, topics, rhythms, seeds, temperatures):
        print('->', ' | '.join(str(s) for s in [artist, topic, rhythm, ' '.join(seed).replace('/ ', ''), temp]))

        batch = {}
        batch['syllables'] = [seed]
        batch['artists'] = [artists * bptt]
        batch['topics'] = [topics * bptt]
        batch['rhythms'] = [rhythms * bptt]

        batch_dict = {}
        for k, vectorizer in vectorizers.items():
            if k == 'targets':
                continue
            batch_dict[k] = vectorizer.transform(batch[k])

        preds = []
        for i in range(max_len):
            pred_proba = model.predict(batch_dict, verbose=0)[0][-1]

            pred_syllab = '<UNK>'
            patience = 20

            while pred_syllab == '<UNK>':
                pred_idx = sample(pred_proba, temp)
                pred_syllab = vectorizers['syllables'].idx2syll[pred_idx]
                patience -= 1

                if patience <= 0:
                    break

            preds.append(pred_syllab)
            batch_dict['syllables'][0] = batch_dict['syllables'][0][1:].tolist() + [pred_idx]


        print(' '.join(preds).replace('/ ', ''))


if __name__ == '__main__':
    main()