import collections
import logging

from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

import lstm
import loaders
import utils

class Scanner:
    def __init__(self, tagger: lstm.Tagger, label_encoder: loaders.Encoder, device="cpu"):
        self.tagger = tagger
        self.label_encoder = label_encoder
        self.device = device
        self.logger = logging.getLogger('Scanner({})'.format(tagger.__class__.__name__))        

    def get_batch(self, batch: Dict[str, List[torch.LongTensor]]) -> Tuple[torch.Tensor]:
        lengths, perm_index = torch.LongTensor(batch['length']).sort(0, descending=True)
        stress = utils.perm_sort(batch['stress'], perm_index)
        wb = utils.perm_sort(batch['wb'], perm_index)
        syllables = utils.perm_sort(batch['syllables'], perm_index)
        
        stress = pad_sequence(stress, batch_first=True).to(self.device)
        wb = pad_sequence(wb, batch_first=True).to(self.device)
        syllables = pad_sequence(syllables, batch_first=True).to(self.device)

        return stress, wb, syllables, lengths, perm_index   

    def scan(self, data: loaders.DataSet):
        results = collections.defaultdict(list)
        n_samples = 0
        with torch.no_grad():
            self.tagger.eval()
            for i, batch in enumerate(data):
                stress, wb, syllables, lengths, perm_index = self.get_batch(batch)
                output = self.tagger(stress, wb, syllables, lengths)
                preds = [self.label_encoder.decode(sample) for sample in output['tags']]
                preds = utils.inverse_perm_sort(preds, perm_index)
                for id, pred in zip(batch['song_id'], preds):
                    results[id].append(pred)
                n_samples += len(preds)
                if n_samples % 500 == 0:
                    self.logger.info(f"Processed {n_samples} samples")
        return results

