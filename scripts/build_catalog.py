import os, glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

import hugs
from hugs.database.tables import Source
from hugs.log import logger
from hugs.utils import ra_dec_to_xyz, angular_dist_to_euclidean_dist
from hugs.utils import euclidean_dist_to_angular_dist


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

    num_sources = session.query(Source).count()
    logger.info('{} sources in raw catalog'.format(num_sources))

    if no_cuts:
        query = session.query(Source)
    else:
        logger.warn('applying cuts')
        query = session.query(Source)\
            .filter(Source.flux_radius_65_g > size_cut_low)\
            .filter(Source.flux_radius_65_g < size_cut_high)\
            .filter(gi > -0.1)\
            .filter(gi < 1.4)\
            .filter(color_line_lo(gi) < gr)\
            .filter(color_line_hi(gi) > gr)

    logger.info('converting query to pandas dataframe')
    cat = pd.read_sql(query.statement, engine)


    hugs_r_e = cat['flux_radius_60_g'] + cat['flux_radius_65_g']
    hugs_r_e *= 0.5
    cat['flux_radius_ave_g'] = hugs_r_e

    hugs_r_e = cat['flux_radius_60_i'] + cat['flux_radius_65_i']
    hugs_r_e *= 0.5
    cat['flux_radius_ave_i'] = hugs_r_e

    hugs_mu_ave = cat['mag_auto_g'].copy()
    hugs_mu_ave += 2.5 * np.log10(2*np.pi*cat['flux_radius_50_g']**2)
    cat['mu_ave_g'] = hugs_mu_ave

    if not no_cuts: 

        mu_cut = (cat['mu_ave_g'] > 22.5) & (cat['mu_ave_g'] < 29.0)
        ell_cut = cat['ellipticity'] < 0.75
        cat = cat[mu_cut & ell_cut]
        logger.info('{} sources in catalog after cuts'.format(len(cat)))

    return cat, session, engine


def remove_duplicates(cat, max_sep=0.2*u.arcsec):
    """
    Remove duplicates in catalog from overlapping patches. 

    Note
    ---- 
    Run this *after* cuts have been made so that it doesn't matter 
    which duplicate measurement we keep. 
    """

    xyz = np.array(ra_dec_to_xyz(cat['ra'], cat['dec'])).T
    kdt = cKDTree(xyz)

    theta = angular_dist_to_euclidean_dist(max_sep.to('deg').value)
    ind = kdt.query_pairs(theta, output_type='ndarray')

    cat.drop(cat.index[ind[:, 1]], inplace=True)


def get_random_subsample(cat, size):

    logger.info('generating random subsample of size = ' + str(size))
    idx = np.random.choice(np.arange(len(cat)), size=size, replace=False)
    subsample = cat.iloc[idx]
    return subsample


if __name__ == '__main__':
    from argparse import ArgumentParser
    default_data_path = '/tigress/jgreco/hsc-s18a/hugs-catalogs'

    parser = ArgumentParser()
    parser.add_argument('--run-name', dest='run_name', required=True)
    parser.add_argument('-o', '--outfile', required=True)
    parser.add_argument('--data-path', dest='data_path',
                        default=default_data_path)
    parser.add_argument('--no-cuts', dest='no_cuts', action='store_true') 
    parser.add_argument('--keep-duplicates', dest='keep_duplicates', 
                        action='store_true') 
    parser.add_argument('--max-sep', dest='max_sep', default=0.2, type=float, 
                        help='count sources as duplicates if they are'
                             'separated by more than this angle in arcsec.')
    parser.add_argument('--random-subsample', default=None, type=int, 
                        dest='random_subsample')
    parser.add_argument('--xmatch-old-cat', dest='xmatch_old_cat', 
                        action='store_true') 
    args = parser.parse_args()

    db_fn = os.path.join(args.data_path, args.run_name)
    db_fn = glob.glob(db_fn + '/*.db')[0]

    logger.info('using database ' + db_fn)
    hugs_cat, session, engine = get_catalog(db_fn, args.no_cuts)

    if not args.keep_duplicates:
        remove_duplicates(hugs_cat, args.max_sep * u.arcsec)
        logger.info(
            '{} sources after removing duplicates'.format(len(hugs_cat)))

    if args.random_subsample is not None:
        hugs_cat = get_random_subsample(hugs_cat, args.random_subsample)

    if args.xmatch_old_cat:
        lsb_cat = Table.read(os.getenv('CAT_1_FN'))
        lsb_sc = SkyCoord(lsb_cat['ra'], lsb_cat['dec'], unit='deg')
        hugs_sc = SkyCoord(hugs_cat['ra'], hugs_cat['dec'], unit='deg')
        _, seps, _ = lsb_sc.match_to_catalog_sky(hugs_sc)
        num_matches = (seps.arcsec < 5.0).sum()
        logger.info('{} out of {} matched with old catalog'.\
                    format(num_matches, len(lsb_cat)))

    hugs_cat = Table.from_pandas(hugs_cat)
    hugs_cat.write(args.outfile, overwrite=True)
