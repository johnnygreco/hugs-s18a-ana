import os
import numpy as np
from astropy.table import Table, vstack

data_path = '/Users/jgreco/local-io/hugs-data/synth-results'

# combine synth results
hugs_match = []
synth_match = []

data = lambda fn: Table.read(os.path.join(data_path, fn))

num_runs = 3
gal_colors = ['blues', 'med', 'reds']

for color in gal_colors:
    for i in range(1, num_runs + 1):
        label = '-{}-0{}.csv'.format(color, i)
        hugs_match.append(data('hugs-match' + label))
        synth_match.append(data('synth-match' + label))

hugs_match = vstack(hugs_match)
synth_match = vstack(synth_match)

num_synths_inject = 5000

mask = (synth_match['mu_e_ave_g'] > 24) & (synth_match['mu_e_ave_g'] < 28)
mask &= np.abs(synth_match['ell'] - 0.3) < 0.05
mask &= (synth_match['r_e'] > 3) & (synth_match['r_e'] < 10)

print('{} synths in catalog'.format(mask.sum()))
