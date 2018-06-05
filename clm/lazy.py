import argparse
import ujson

def get_lazy(oldfile, newfile):
    with open(oldfile, 'r') as f:
        songs = ujson.load(f)

    with open(newfile, 'w') as f:
        for song in songs:
            f.write(ujson.dumps(song) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generation of lazy file')
    parser.add_argument('--oldfile', type=str, default='../data/ohhla-beatstress.json',
                        help='path to incoming data file (unlazy json)')
    parser.add_argument('--newfile', type=str, default='../data/lazy_ohhla-beatstress.json',
                        help='path to outgoing data file (lazy json)')
    args = parser.parse_args()

    get_lazy(args.oldfile, args.newfile)


if __name__ == '__main__':
    main()