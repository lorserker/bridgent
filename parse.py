import sys
import glob
import gzip
import os.path

import numpy as np

card_index_lookup = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        range(13)
    )
)

def set_data(X, i, deal_str):
    hands = deal_str.split('\t')
    assert(len(hands) == 4)

    for hand_index in [0, 1, 2, 3]:
        assert(len(hands[hand_index]) == 20)
        suits = hands[hand_index].split()
        assert(len(suits) == 4)
        for suit_index in [0, 1, 2, 3]:
            for card in suits[suit_index][1:]:
                card_index = card_index_lookup[card]
                X[i, suit_index, card_index, hand_index] = 1

def create_binary(n, raw_dir, bin_dir):
    X = np.zeros((n, 4, 13, 4), dtype=np.uint8)
    y_notrump = np.zeros((n, 1), dtype=np.uint8)
    y_spades = np.zeros((n, 1), dtype=np.uint8)

    k = 0
    for fnm in sorted(glob.glob(os.path.join(raw_dir, '*.gz'))):
        print(fnm)
        for line in gzip.open(fnm):
            i = k // 3
            mod = k % 3
            if mod == 0:
                deal_str = line.decode('ascii').strip()
                set_data(X, i, deal_str)
            elif mod == 1:
                y_notrump[i] = int(line.decode('ascii').split()[0])
            else:
                y_spades[i] = int(line.decode('ascii').split()[0])
            k += 1

    np.save(os.path.join(bin_dir, 'deal.npy'), X)
    np.save(os.path.join(bin_dir, 'tricks_notrump.npy'), y_notrump)
    np.save(os.path.join(bin_dir, 'tricks_spades.npy'), y_spades)


if __name__ == '__main__':
    n = int(sys.argv[1])
    raw_dir = sys.argv[2]
    bin_dir = sys.argv[3]

    create_binary(n, raw_dir, bin_dir)
