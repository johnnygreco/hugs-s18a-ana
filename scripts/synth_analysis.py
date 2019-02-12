
import os, glob
import hugs
import numpy as np
import pandas as pd
from hugs.database.tables import Source, Synth
from hugs.log import logger
from astropy.table import Table
from collections import namedtuple
from astropy.coordinates import SkyCoord
from astropy import units as u
from utils import labels, project_dir


def match_synths(hugs_cat, synth_cat, min_sep=1.0*u.arcsec):
        
    if type(hugs_cat) != Table:
        hugs_cat = Table.from_pandas(hugs_cat)
    if type(synth_cat) != Table:
        synth_cat = Table.from_pandas(synth_cat)

    hugs_coord = SkyCoord(hugs_cat['ra'], hugs_cat['dec'], unit='deg')
    synth_coord = SkyCoord(synth_cat['ra'], synth_cat['dec'], unit='deg')

    logger.info('finding nearest neighbor within {:.1f}'.\
                format(min_sep))
    hugs_idx_1, sep_1, _ = synth_coord.match_to_catalog_sky(hugs_coord, 1)
    synth_mask_1 = sep_1 < min_sep
    synth_match = synth_cat[synth_mask_1]

    logger.info('finding second nearest neighbor within {:.1f}'.\
                format(min_sep))
    hugs_idx_2, sep_2, _ = synth_coord.match_to_catalog_sky(hugs_coord, 2)
    synth_mask_2 = sep_2 < min_sep

    synth_mask_12 = synth_mask_1 & synth_mask_2
    synth_match_12 = synth_cat[synth_mask_12]

    hugs_match_1 = hugs_cat[hugs_idx_1][synth_mask_12]
    hugs_match_2 = hugs_cat[hugs_idx_2][synth_mask_12]

    logger.info('keeping match with better radius measurement')
    diff_1 = np.abs(synth_match_12['r_e'] - hugs_match_1['flux_radius_50_i'])
    diff_2 = np.abs(synth_match_12['r_e'] - hugs_match_2['flux_radius_50_i'])
    switch_match = diff_1 > diff_2

    logger.warn('switching {} matches'.format(switch_match.sum()))
    mask_idx = np.argwhere(synth_mask_12)[~switch_match]
    synth_mask_12[mask_idx] = False

    hugs_idx_1[synth_mask_12] = hugs_idx_2[synth_mask_12]
    hugs_match = hugs_cat[hugs_idx_1[synth_mask_1]]

    return hugs_match, synth_match


def completeness_grids(cat):
    pass


def parameter_accuracy(hugs_cat, synth_cat, fig_dir):
    pass




def get_catalog(db_fn, no_cuts=False):

    logger.info('connecting to hugs database')
    engine = hugs.database.connect(db_fn)
    session = hugs.database.Session()

    size_cut_low = 2.5
    size_cut_high = 100.0
    m, b = 0.7, 0.4

    color_line_lo =  lambda _x: m*_x - b
    color_line_hi =  lambda _x: m*_x + b
    gi = Source.mag_ap9_g - Source.mag_ap9_i 
    gr = Source.mag_ap9_g - Source.mag_ap9_r 

    if no_cuts:
        query = session.query(Source)
    else:
        logger.warn('applying cuts')
        query = session.query(Source)\
            .filter(Source.flux_radius_60_i > size_cut_low)\
            .filter(Source.flux_radius_60_i < size_cut_high)\
            .filter(gi > -0.1)\
            .filter(gi < 1.4)\
            .filter(color_line_lo(gi) < gr)\
            .filter(color_line_hi(gi) > gr)

    logger.info('converting query to pandas dataframe')
    cat = pd.read_sql(query.statement, engine)

    return cat


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    default_synth_dir = '/tigress/jgreco/hsc-s18a/synths/global'
    parser = ArgumentParser()
    parser.add_argument('--run-name', dest='run_name', required=True)
    parser.add_argument('--synth-dir', dest='synth_dir', 
                        default=default_synth_dir)
    parser.add_argument(
        '--synth-cat-fn', dest='synth_cat_fn', 
        default=os.path.join(default_synth_dir, 'global-synth-cat.fits'))
    parser.add_argument('--fig-dir', dest='fig_dir', 
                        default=os.path.join(project_dir, 'figs'))
    parser.add_argument('--no-cuts', dest='no_cuts', action='store_true')
    parser.add_argument('--cat-fn', dest='cat_fn', default=None)
    parser.add_argument('--save-fn', dest='save_fn', default=None)
    parser.add_argument('--min-sep', dest='min_sep', default=1.0, type=float)
    args = parser.parse_args()

    if args.cat_fn is None:
        db_fn = os.path.join(args.synth_dir, args.run_name)
        db_fn = glob.glob(db_fn + '/*.db')[0]
        logger.info('using database ' + db_fn)
        hugs_cat = get_catalog(db_fn, args.no_cuts)
        if args.save_fn is not None:
            logger.info('saving hugs catalog to ' + args.save_fn)
            hugs_cat.to_csv(args.save_fn)
    else:
        hugs_cat = pd.read_csv(args.cat_fn)

    synth_cat = Table.read(args.synth_cat_fn)


    hugs_match, synth_match = match_synths(
        hugs_cat, synth_cat, min_sep=args.min_sep*u.arcsec)


    parameter_accuracy(hugs_match, synth_match, args.fig_dir)





