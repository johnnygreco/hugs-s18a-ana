import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from hugs.utils import mkdir_if_needed
import lsstutils

parser = ArgumentParser()
parser.add_argument('-c', '--cat-fn', dest='cat_fn', required=True)
parser.add_argument('-o', '--outdir', required=True)
args = parser.parse_args()

mkdir_if_needed(args.outdir)






