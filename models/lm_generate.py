
import seqmod.utils as u


def make_reducer(sep=''):
    def func(line):
        return sep.join(line)
    return func


def syll_reducer(line):
    output = ''
    for syl in line:
        if output.endswith('-'):
            if syl.startswith('-'):
                output = output[:-1] + syl[1:]
            else:
                output = output[:-1] + ' ' + syl
                print("oops")
        else:
            if syl.startswith('-'):
                output += syl[1:]
                print("oops")
            else:
                output += ' ' + syl

    return output

def run_generate(model, reducer, batch_size, max_seq_len, temperature):
    scores, hyps = model.generate(
        model.embeddings.d, batch_size=batch_size, max_seq_len=max_seq_len,
        ignore_eos=True, temperature=temperature)

    for hyp in hyps:
        line = []
        song = []
        for sym in hyp:
            if sym == model.embeddings.d.get_eos():
                song.append(reducer(line))
                line = []
            else:
                line.append(model.embeddings.d.vocab[sym])
        yield song


def write_output(songs, outputfile, n_sents):
    sents = 0
    with open(outputfile, 'w+') as f:
        for song in songs:
            for line in song:
                if sents >= n_sents:
                    break
                f.write(line)
                f.write('\n')
                sents += 1
            f.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--n_sents', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--outputfile', default='generated.txt')
    args = parser.parse_args()

    model = u.load_model(args.model)
    sents_batch = max(int(args.n_sents / args.batch_size), 1)

    if len(model.embeddings.d) < 1000:  # char level: 100 chars/sent
        reducer = make_reducer()
        max_seq_len = sents_batch * 100
    else:                       # syll level: 20 syll/sent
        reducer = make_reducer(' ')
        max_seq_len = sents_batch * 20

    songs = run_generate(model, reducer, args.batch_size,
                         max_seq_len=max_seq_len, temperature=args.temperature)

    write_output(songs, args.outputfile, args.n_sents)
