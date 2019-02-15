"""
Script for processing batch of images in a directory into a single figure (image+log)

Run with --help argument to see usage.
"""
import os
import pathlib
import argparse
from glob import glob
import pandas as pd

from coremdlr.facies.models import PredictorModel

DEFAULT_TRAIN_PATH = '/home/'+os.environ['USER']+'/Dropbox/core_data/train_data/'


def make_preds_csv(args):
    model = PredictorModel(None, None, {'from_file': args.model})

    fdset =  FaciesDataset



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('well',
        type=str,
        help="Name of (schiehallion-complex) well [change `DEFAULT_TRAIN_PATH` if not in S-C]."
    )
    parser.add_argument('--model',
        type=str,
        default=None,
        help='Path to saved PredictorModel to load and use for prediction.'
    )
    parser.add_argument('--save_csv',
        type=str,
        default=None,
        help="Name of the csv file to save. default=\'<well>_preds.csv\'"
    )

    args = parser.parse_args()

    assert args.model is not None, 'Must specify a PredictorModel to load.'

    make_preds_csv(args)


if __name__ == '__main__':
    main()
