import argparse
import math
import os
import time

import numpy as np
import torch
import models


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--emb_dim', type=int, default=32,
                    help='size of word embeddings')
parser.add_argument('--hid_dim', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--n_layers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--quantization', type=int, default=16,
                    help='Divide each bar into Q ticks.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midi_encoder = models.MidiEncoder(q=args.quantization)
data = models.DrumData(args.data, midi_encoder)
rnn = models.LSTM(
    input_size=len(midi_encoder.index),
    hidden_size=args.hid_dim,
    emb_dim=args.emb_dim,
    num_classes=len(midi_encoder.index),
    n_layers=args.n_layers,
    dropout=args.dropout
).to(device)

lr = args.lr
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', verbose=True, patience=5)
best_val_loss = float("inf")


def train():
    rnn.train()
    epoch_loss = 0.
    start_time = time.time()
    for i, (inputs, targets, lengths) in enumerate(data.train_batches(), 1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, _ = rnn(inputs, lengths)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(2)), targets.view(-1),
            ignore_index=0)
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), args.clip)
        epoch_loss += loss.item()
        optimizer.step()
        if i % 5 == 0:
            curr_loss = epoch_loss / i
            elapsed = time.time() - start_time
            print(f'| epoch {epoch} | lr {lr:02.4f} | time {elapsed:5.2f} | '
                  f'loss {curr_loss:5.2f} | ppl {math.exp(curr_loss):8.2f}')
            start_time = time.time()

def evaluate():
    rnn.eval()
    val_loss, n_batches = 0., 0
    with torch.no_grad():
        for inputs, targets, lengths in data.dev_batches():
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = rnn(inputs, lengths)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(2)), targets.view(-1),
                ignore_index=0)
            val_loss += loss.item()
            n_batches += 1
    return val_loss / n_batches
                
try:
    for epoch in range(args.epochs):
        start_time = time.time()
        train()
        val_loss = evaluate()
        print('- ' * 45)
        print(f'| end of epoch {epoch:3d} | time: {time.time() - start_time:5.2f}s '
              f'| valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f}')
        if epoch % 10 == 0:
            samples = models.sample_from_piano_rnn(
                rnn, sample_length=args.quantization, temperature=1)
            midi_encoder.sequence_to_midi(
                midi_encoder.decode_sequence(samples)).write(f'sample-{epoch}.mid')
        if not best_val_loss or val_loss < best_val_loss:
            with open('beats-model.pth', 'wb') as f:
                torch.save(rnn, f)
            best_val_loss = val_loss
        scheduler.step(val_loss)
except KeyboardInterrupt:
    print('- ' * 45)
    print('Exiting from training early...')
    
