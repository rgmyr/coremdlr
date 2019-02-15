import numpy as np
import pandas as pd

from operator import attrgetter
from collections import namedtuple

from xml.etree import ElementTree


class Section():
    pass


def xml2array(fpath):

    tree = ElementTree.parse(fpath)
    height = eval(tree.find('size').find('height').text)
    label_array = np.zeros((height,), dtype='a2')

    labels = []
    ymins = []
    for obj in tree.findall('object'):
        labels.append(obj.find('name').text)
        bbox = obj.find('bndbox')
        ymins.append(eval(bbox.find('ymin').text))

    ymaxs = 
    idxs = np.argsort(ymins)



    for :
         = 0 if i==0 else section[]

    return sections


def xml2dict(fpath):
    '''Convert XML object labels file to a dict of {label: [start_row, end_row]}.'''
    tree = ElementTree.parse(fpath)

    label_dict = {obj.find('name').text: [] for obj in tree.findall('object')}

    for obj in tree.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        label_dict[label].append( (eval(bbox.find('ymin').text), eval(bbox.find('ymax').text)) )

    return label_dict


def xml2list(fpath, collapse=False, tolerance=15):
    """
    Convert XML object labels file to flat list of (start_row, end_row).
        - if collapse=True, assumes start/end values within <tolerance> of each other
            are meant to be the same boundary
    """
    tree = ElementTree.parse(fpath)

    pairs = []

    for obj in tree.findall('object'):
        bbox = obj.find('bndbox')
        pairs.append( (eval(bbox.find('ymin').text), eval(bbox.find('ymax').text)) )

    if collapse:
        colap = sorted(list(sum(pairs, ())))
        return np.array(colap)[np.where(np.diff(colap)>tolerance)].tolist()
    else:
        return sorted(pairs)



if __name__ == '__main__':
    fpath = '/home/administrator/code/python/coremdlr/coremdlr/facies/datasets/apache/apache_7882.0-7932.0_labels.xml'
    print(xml2array(fpath))
