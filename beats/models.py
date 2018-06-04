import collections
import math
import os
import numpy as np
from typing import List, Tuple, Set

import pretty_midi
import sklearn.model_selection

import torch
from torch.nn.utils.rnn import pad_sequence



DEFAULT_DRUM_TYPE_PITCHES = [
    # bass drum
    [36, 35],
    # snare drum
    [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85],
    # closed hi-hat
    [42, 44, 54, 68, 69, 70, 71, 73, 78, 80],
    # open hi-hat
    [46, 67, 72, 74, 79, 81],
    # low tom
    [45, 29, 41, 61, 64, 84],
    # mid tom
    [48, 47, 60, 63, 77, 86, 87],
    # high tom
    [50, 30, 43, 62, 76, 83],
    # crash cymbal
    [49, 55, 57, 58],
    # ride cymbal
    [51, 52, 53, 59, 82]
]


class DrumEncoding:
    def __init__(self, drum_pitches=None, ignore_unknown_drums=True):
        if drum_pitches is None:
            drum_pitches = DEFAULT_DRUM_TYPE_PITCHES
        self._drum_map = {i: pitches for i, pitches in enumerate(drum_pitches)}
        self._inverse_drum_map = {pitch: index
                                  for index, pitches in self._drum_map.items()
                                  for pitch in pitches}
        self._ignore_unknown_drums = ignore_unknown_drums

    def __len__(self):
        return len(self._drum_map)

    def encode(self, pitch: int) -> int:
        return self._inverse_drum_map[pitch]

    def decode(self, string: str) -> Set[int]:
        elts = [int(e) for e in string]
        return {self._drum_map[i][0] for i, d in enumerate(elts) if d > 0}


class MidiEncoder:
    def __init__(self, pad_symbol: str = '<PAD>', q: int = 16) -> None:
        self.pad_symbol = pad_symbol
        self.index = collections.defaultdict()
        self.index.default_factory = lambda: len(self.index)
        self.index[pad_symbol]
        self.encoder = DrumEncoding()
        self.q = q

    def encode_sequence(self, sequence: List[List[int]]) -> Tuple[torch.tensor, torch.tensor, int]:
        strings = [self.index['<START>']]
        for i, row in enumerate(sequence):
            if i > 0 and i % self.q == 0:
                strings.append(self.index['<BAR>'])
            strings.append(self.index[''.join(map(str, row))])
        strings.append(self.index['<END>'])
        return (torch.LongTensor(strings[:-1]), torch.LongTensor(strings[1:]), len(strings) - 1)

    def encode_midi(self, midi_file: str) -> Tuple[torch.tensor, torch.tensor, int]:
        return self.encode_sequence(self.midi_to_sequence(midi_file))

    def midi_to_sequence(self, midi_file: str) -> None:
        stream = pretty_midi.PrettyMIDI(midi_file)
        tempo = stream.get_tempo_changes()[1][0]
        beats = np.arange(0, stream.get_end_time(), 60 / tempo / (self.q / 4))
        encoding = np.zeros(
            (int(math.ceil(beats.size / self.q)) * self.q, len(self.encoder)), dtype=int)
        for instrument in stream.instruments:
            for note in instrument.notes:
                idx = np.abs(note.start - beats).argmin()
                encoding[idx, self.encoder.encode(note.pitch)] = 1
        return encoding

    def decode_sequence(self, sequence):
        if not hasattr(self, 'index2item'):
            self.index2item = sorted(self.index, key=self.index.get)
        return [self.index2item[elt] for elt in sequence]

    def sequence_to_midi(self, sequence, bpm: int = 80) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        instrument = pretty_midi.Instrument(0)
        pm.instruments.append(instrument)
        time_signature = pretty_midi.containers.TimeSignature(4, 4, 4)
        pm.time_signature_changes.append(time_signature)
        instrument_events = collections.defaultdict(
            lambda: collections.defaultdict(list))
        time = 0
        tick = 60 / bpm / (self.q / 4)
        for i, encoding in enumerate(sequence, 1):
            if encoding not in ('<START>', '<BAR>', '<END>', '<PAD>'):
                encoding = self.encoder.decode(encoding)
                if encoding: # empty means tick rest
                    for pitch in encoding:
                        instrument_events[(pitch, 9, True)]['notes'].append(
                            pretty_midi.Note(100, pitch, time,  time + tick))
                time += tick
        for (instr_id, prog_id, is_drum) in sorted(instrument_events.keys(), reverse=True):
            if instr_id > 0:
                instrument = pretty_midi.Instrument(
                    prog_id, is_drum, name=pretty_midi.note_number_to_drum_name(instr_id))
                pm.instruments.append(instrument)
            instrument.program = prog_id
            instrument.notes = instrument_events[
                (instr_id, prog_id, is_drum)]['notes']
        return pm
    

def perm_sort(x, index):
    return [x[i] for i in index]


class DrumData:
    def __init__(self, midi_dir: str, encoder: MidiEncoder) -> None:
        self.midi_files = [encoder.encode_midi(f.path) for f in os.scandir(midi_dir)]
        self.train, self.dev = sklearn.model_selection.train_test_split(
            self.midi_files, test_size=0.1, shuffle=True)

    def train_batches(self, batch_size=20):
        return self.get_batches(self.train, batch_size=batch_size)

    def dev_batches(self, batch_size=20):
        return self.get_batches(self.dev, batch_size=batch_size)

    def get_batches(self, midi_files, batch_size=20):
        inputs, targets, lengths, i = [], [], [], 0
        for input, target, length in midi_files:
            inputs.append(input); targets.append(target); lengths.append(length)
            i += 1
            if i == batch_size:
                yield self.post_batch(inputs, targets, lengths)
                inputs, targets, lengths, i = [], [], [], 0
        if inputs:
            yield self.post_batch(inputs, targets, lengths)

    def post_batch(self, inputs, targets, lengths):
        lengths, perm_index = torch.LongTensor(lengths).sort(0, descending=True)
        inputs = perm_sort(inputs, perm_index)
        targets = perm_sort(targets, perm_index)
        inputs = pad_sequence(inputs, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        return inputs, targets, lengths


def sample_from_piano_rnn(rnn, sample_length=16, temperature=1, start=None):
    if start is None:
        current_input = torch.zeros(1, 1, dtype=torch.int).type(torch.LongTensor)
        current_input[0, 0] = 1
        current_input = current_input.to(device)
    hidden = None
    final_output = [current_input.data.squeeze(1)]
    for i in range(sample_length):
        output, hidden = rnn(current_input, [1], hidden)
        probabilities = torch.nn.functional.log_softmax(output.div(temperature), dim=2).exp()
        current_input = torch.multinomial(
            probabilities.view(-1), 1).squeeze().unsqueeze(0).unsqueeze(1)
        final_output.append(current_input.data.squeeze(1))
    sampled_sequence = torch.cat(final_output, dim=0).cpu().numpy()
    return sampled_sequence


class LSTM(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, emb_dim:
                 int, num_classes: int, n_layers: int = 2, dropout: float = 0.5) -> None:
        super(LSTM, self).__init__()
        self.note_encoder = torch.nn.Embedding(num_classes, emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(emb_dim, hidden_size, n_layers, batch_first=True)
        self.projection_layer = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = dropout

    def forward(self, sequences, lengths, hidden=None):
        embs = self.note_encoder(sequences)
        embs = torch.nn.functional.dropout(embs, p=self.dropout, training=self.training)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embs, lengths, batch_first=True)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        logits = self.projection_layer(outputs)
        return logits, hidden

if __name__ == '__main__':
    midi_encoder = MidiEncoder(q=24)
    data = DrumData('midifiles', midi_encoder)
    rnn = LSTM(input_size=len(midi_encoder.index),
               hidden_size=512,
               emb_dim=32,
               num_classes=len(midi_encoder.index),
               n_layers=3,
               dropout=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn.to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, patience=5)

    clip = 1.0
    epochs = 100
    best_val_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (inputs, targets, lengths) in enumerate(data.train_batches(), 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits, _ = rnn(inputs, lengths)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(2)), targets.view(-1),
                size_average=True, ignore_index=0)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm(rnn.parameters(), clip)
            optimizer.step()
            if i % 5 == 0:
                print("Epoch {}, batch {}, Loss {}".format(epoch, i, epoch_loss / i))
            # validate
        with torch.no_grad():
            rnn.eval()
            val_loss = 0.0
            n_batches = 0
            for inputs, targets, lengths in data.dev_batches():
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits, _ = rnn(inputs, lengths)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(2)), targets.view(-1),
                    size_average=True, ignore_index=0)
                val_loss += loss.item()
                n_batches += 1
            val_loss = val_loss / n_batches
            print("Validation loss: {}".format(val_loss))
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                torch.save(rnn.state_dict(), 'music_rnn.pth')
                best_val_loss = val_loss
            if epoch % 10 == 0:
                samples = sample_from_piano_rnn(rnn, sample_length=24, temperature=1)
                midi_encoder.sequence_to_midi(
                    midi_encoder.decode_sequence(samples)).write(f'output-{epoch}.mid')
            rnn.train()
