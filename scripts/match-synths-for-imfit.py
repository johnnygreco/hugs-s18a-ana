import os, glob
from argparse import ArgumentParser

import numpy as np
from astropy.table import Table
from astropy import units as u

from synth_analysis import _get_unique_synths, match_synths
from synth_analysis import get_injected_synth_ids 
from build_catalog import get_catalog
from hugs.database.tables import Source, Synth
from hugs.log import logger
from utils import param_dict, labels, project_dir

default_synth_dir = '/tigress/jgreco/hsc-s18a/synths/global'
parser = ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--run-name', dest='run_name')
group.add_argument('--cat-fn', dest='cat_fn', default=None)

parser.add_argument('-o', '--out-fn', dest='out_fn', required=True)
parser.add_argument('--min-sep', dest='min_sep', default=1.0, type=float)
parser.add_argument('--synth-dir', dest='synth_dir',
                    default=default_synth_dir)
parser.add_argument(
    '--synth-cat-fn', dest='synth_cat_fn',
    default=os.path.join(default_synth_dir, 'global-synth-cat.fits'))
parser.add_argument('--save-hugs-fn', dest='save_fn', default=None)
parser.add_argument('--no-cuts', dest='no_cuts', action='store_true')
args = parser.parse_args()


if args.cat_fn is None:
    db_fn = os.path.join(args.synth_dir, args.run_name)
    db_fn = glob.glob(db_fn + '/*.db')[0]
    logger.info('using database ' + db_fn)
    hugs_cat, session, engine = get_catalog(db_fn, args.no_cuts)
    synth_ids = get_injected_synth_ids(session=session, engine=engine)
    if args.save_fn is not None:
        logger.info('saving hugs catalog to ' + args.save_fn)
        hugs_cat.to_csv(args.save_fn)
        fn = args.save_fn[:-4] + '-synth-ids.csv'
        logger.info('saving synth id catalog to ' + fn)
        synth_ids.to_csv(fn)
else:
    hugs_cat = pd.read_csv(args.cat_fn)
    fn = args.cat_fn[:-4] + '-synth-ids.csv'
    synth_ids = pd.read_csv(fn)


synth_cat = Table.read(args.synth_cat_fn)
synth_id_unique, injected_synth = _get_unique_synths(synth_ids)
masked = injected_synth['mask_bright_object'].values.astype(bool)
masked |= injected_synth['mask_no_data'].values.astype(bool)
masked |= injected_synth['mask_sat'].values.astype(bool)

synth_cat = synth_cat[synth_id_unique - 1]
synth_cat = synth_cat[~masked]

logger.info('matching hugs and synth catalogs')

hugs_match, synth_match = match_synths(
    hugs_cat, synth_cat, min_sep=args.min_sep*u.arcsec)

hugs_match['synth_id'] = synth_match['synth_id']

logger.info('matched hugs cat has {} sources'.format(len(hugs_match)))

logger.info('writing to ' + args.out_fn)
hugs_match.write(args.out_fn, overwrite=True) 
