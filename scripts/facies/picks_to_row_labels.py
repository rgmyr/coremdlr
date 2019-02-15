"""
Script for processing pair(s) of picks.csv + depth.npy files to generate row-wise labels.npy.

Run with --help argument to see usage.
"""
import os
import argparse
from glob import glob
import numpy as np
import pandas as pd

def common_path(path_list):
    '''Common prefix of all paths in path_list.'''
    chars = []
    for tup in zip(*path_list):
        if len(set(tup)) == 1:
            chars.append(tup[0])
        else:
            break
    return ''.join(chars)


def picks_to_rows(picks_file, depth_file):
    '''Take a picks and depth file, save the row-wise labels .npy file.'''
    picks = pd.read_csv(picks_file, usecols=['top', 'base', 'lithology'])
    depth = np.load(depth_file)
    row_labels = np.zeros_like(depth, dtype='a2')    # labels stored as 2-char strings

    idx = 0
    current_pick = picks.iloc[idx]
    for i in range(row_labels.size):
        if depth[i] > current_pick['base']:
            idx += 1
            current_pick = picks.iloc[idx]
        row_labels[i] = current_pick.lithology

    save_path = common_path([picks_file, depth_file])+'labels.npy'
    print('Saving to:', save_path)
    np.save(save_path, row_labels)


def convert_picks_files(args):
    picks_files = sorted(glob(os.path.join(args.dir, args.wells_prefix + '*_picks.csv')))
    depth_files = sorted(glob(os.path.join(args.dir, args.wells_prefix + '*_depth.npy')))

    assert len(picks_files) == len(depth_files), 'Must have the same number of picks and depth files'
    assert len(picks_files) > 0, 'Must have at least one pair of picks + depth files'

    for p_file, d_file in zip(picks_files, depth_files):
        print('\nGenerating row labels for pair of files:')
        print(os.path.split(p_file)[-1], os.path.split(d_file)[-1])
        picks_to_rows(p_file, d_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
        type=str,
        help="Path to directory containing target _depth.npy and _picks.csv files."
    )
    parser.add_argument('--wells_prefix',
        dest='wells_prefix',
        type=str,
        default='',
        help="Prefix of target *_depth.npy and *_picks.csv files, default=''."
    )

    convert_picks_files(parser.parse_args())


if __name__ == '__main__':
    main()
