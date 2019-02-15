"""
Check the pipeline completion times for patches using 
the hugs log files. The log directory is given as a 
command-line argument.
"""

import os
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('path')
parser.add_argument('--print-running-patches', 
                    dest='print_running_patches',
                    action='store_true')
args = parser.parse_args()

os.chdir(args.path)

files = [f for f in os.listdir('.') if 'log' in f]

times = []
count = 0
running = 0
no_run = 0
no_sources = 0

for fn in files:
    with open(fn, 'r') as f:
        lines = f.readlines()
        complete = False
        for line in lines:
            if line.startswith('task completed'):
                times.append(float(line.split()[-2]))
                count += 1
                complete = True
            elif line.startswith('***** not enough data'):
                complete = True
                count += 1
                no_run += 1
            elif line.startswith('**** no sources detected'):
                complete = True
                count += 1
                no_sources += 1
            elif line.startswith('**** no matched sources'):
                complete = True
                count += 1
                no_sources += 1
            elif line.startswith('**** no sources found'):
                complete = True
                count += 1
                no_sources += 1
        if not complete:
            running += 1
            if args.print_running_patches:
                print(fn)



times = np.array(times)

print('\n{} completed patches\n'.format(count))
print('\n{} running patches\n'.format(running))
print('\n{} patches without enough data\n'.format(no_run))
print('\n{} patches with no sources\n'.format(no_sources))
print('completion time statistics')
print('--------------------------')
print('mean   = {:.2f}'.format(times.mean()))
print('median = {:.2f}'.format(np.median(times)))
print('stddev = {:.2f}'.format(times.std()))
print('min    = {:.2f}'.format(times.min()))
print('max    = {:.2f}'.format(times.max()))
print('--------------------------')
