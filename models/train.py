import argparse

import torch

from lstm import LSTMTagger, CRFLSTMTagger, CRFTagger, Trainer, embedding_layer
import loaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tagger', default='lstm', choices=('lstm', 'crf-lstm', 'crf'))
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--train_file', required=True, type=str)
    parser.add_argument('--dev_file', required=True, type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--pretrained_embeddings', required=True, type=str)
    parser.add_argument('--finetune_embeddings', action='store_true')
    args = parser.parse_args()
    
    syllable_vocab, syllable_vectors = loaders.load_gensim_embeddings(args.pretrained_embeddings)

    eos, bos = None, None
    if args.tagger in ('crf-lstm', 'crf'):
        eos, bos = '<EOS>', '<BOS>'
    stress_encoder = loaders.Encoder('stress', preprocessor=loaders.normalize_stress,
                                     eos_token=eos, bos_token=bos)
    beat_encoder = loaders.Encoder('beatstress', preprocessor=loaders.normalize_stress,
                                   unk_token=None, bos_token=bos, eos_token=eos)
    syllable_encoder = loaders.Encoder('syllables', vocab=syllable_vocab, fixed_vocab=True,
                                       eos_token=eos, bos_token=bos)
    wb_encoder = loaders.Encoder('syllables', preprocessor=loaders.word_boundaries,
                                 eos_token=eos, bos_token=bos)

    syllable_embeddings = embedding_layer(
        syllable_vectors, n_padding_vectors=len(syllable_encoder.reserved_tokens),
        padding_idx=syllable_encoder.pad_index, trainable=args.finetune_embeddings)

    train = loaders.DataSet(args.train_file, stress=stress_encoder, beatstress=beat_encoder,
                            wb=wb_encoder, syllables=syllable_encoder, batch_size=args.batch_size)
    dev = loaders.DataSet(args.dev_file, stress=stress_encoder, beatstress=beat_encoder,
                          wb=wb_encoder, syllables=syllable_encoder, batch_size=args.batch_size)
    if args.test_file is not None:
        test = loaders.DataSet(args.test_file, stress=stress_encoder, beatstress=beat_encoder,
                               wb=wb_encoder, syllables=syllable_encoder, batch_size=args.batch_size)    
    # construct a layer holding the pre-trained word2vec embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tagger = LSTMTagger if args.tagger == 'lstm' else CRFLSTMTagger if args.tagger == 'crf-lstm' else CRFTagger
    # Initialize the tagger
    tagger = tagger(
        args.emb_dim, args.hid_dim, args.num_layers, args.dropout, stress_encoder.size() + 2,
        wb_encoder.size() + 2, syllable_embeddings, beat_encoder.size() + 2,
        args.batch_size, bidirectional=True)
    print(tagger)
    # The Adam optimizer seems to be working fine. Make sure to exlcude the pretrained
    # embeddings from optimizing
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, tagger.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay
    ) # RMSprop
    tagger.to(device)
    trainer = Trainer(
        tagger, train, dev, test, optimizer, device=device)
    trainer.train(epochs=args.epochs)
    with torch.no_grad():
        tagger.eval() # set all trainable attributes to False
        trainer.test(statefile='model_best.pth.tar')
        tagger.train() # set all trainable attributes back to True
