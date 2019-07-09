import os, glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from utils import load_nsa

import hugs
from hugs.database.tables import Source, Tract, Patch
from hugs.log import logger
from hugs.utils import ra_dec_to_xyz, angular_dist_to_euclidean_dist


def get_catalog(db_fn, no_cuts=False, morph_cut=True, no_ext=False, 
                nsa_cut=False, nsa_min_mass=1e10, nsa_rad_factor=5, 
                cirrus_cut=False, max_sources=50, max_A_g=0.3):

    logger.info('connecting to hugs database')
    engine = hugs.database.connect(db_fn)
    session = hugs.database.Session()

    size_cut_low = 2.5
    size_cut_high = 1000.0
    m, b = 0.7, 0.4

    color_line_lo =  lambda _x: m*_x - b
    color_line_hi =  lambda _x: m*_x + b


    if not no_ext:
        logger.warning('applying extinction correction to cut parameters')

    A_g = 0.0 if no_ext else Source.A_g 
    A_r = 0.0 if no_ext else Source.A_r
    A_i = 0.0 if no_ext else Source.A_i

    gi = Source.mag_ap9_g - Source.mag_ap9_i - (A_g - A_i)
    gr = Source.mag_ap9_g - Source.mag_ap9_r - (A_g - A_r)

    num_sources = session.query(Source).count()
    logger.info('{} sources in raw catalog'.format(num_sources))

    if no_cuts:
        query = session.query(Source)
        query = session.query(Tract.hsc_id.label('tract'), 
                              Patch.hsc_id.label('patch'), Source).\
                              join(Patch, Patch.tract_id==Tract.id).\
                              join(Source, Source.patch_id==Patch.id)
    else:
        logger.warning('applying cuts')
        query = session.query(Tract.hsc_id.label('tract'), 
                              Patch.hsc_id.label('patch'), Source).\
                              join(Patch, Patch.tract_id==Tract.id).\
                              join(Source, Source.patch_id==Patch.id).\
                              filter(Source.flux_radius_65_g > size_cut_low).\
                              filter(Source.flux_radius_65_g < size_cut_high).\
                              filter(Source.flux_radius_50_g > 0).\
                              filter(Source.flux_radius_50_r > 0).\
                              filter(Source.flux_radius_50_i > 0).\
                              filter(gi > -0.1).\
                              filter(gi < 1.4).\
                              filter(color_line_lo(gi) < gr).\
                              filter(color_line_hi(gi) > gr)

    logger.info('converting query to pandas dataframe')
    cat = pd.read_sql(query.statement, engine)

    hugs_r_e = cat['flux_radius_60_g'] + cat['flux_radius_65_g']
    hugs_r_e *= 0.5
    cat['flux_radius_ave_g'] = hugs_r_e

    hugs_r_e = cat['flux_radius_60_i'] + cat['flux_radius_65_i']
    hugs_r_e *= 0.5
    cat['flux_radius_ave_i'] = hugs_r_e

    A_g = 0.0 if no_ext else cat['A_g']
    A_r = 0.0 if no_ext else cat['A_r']
    A_i = 0.0 if no_ext else cat['A_i']
    
    hugs_mu_ave = cat['mag_auto_i'].copy()
    hugs_mu_ave += 2.5 * np.log10(2*np.pi*cat['flux_radius_50_i']**2)
    cat['mu_ave_i'] = hugs_mu_ave - A_i

    hugs_mu_ave = cat['mag_auto_g'].copy()
    hugs_mu_ave += 2.5 * np.log10(2*np.pi*cat['flux_radius_50_g']**2)
    cat['mu_ave_g'] = hugs_mu_ave - A_g

    cat['g-i'] = cat['mag_ap9_g'] - cat['mag_ap9_i'] - (A_g - A_i)
    cat['g-r'] = cat['mag_ap9_g'] - cat['mag_ap9_r'] - (A_g - A_r)

    # HACK: not sure why the tracts aren't integers
    tracts = []
    for t in cat['tract']:
        tracts.append(np.frombuffer(t, np.int64)[0])
    cat['tract'] = tracts

    if not no_cuts: 

        mu_cut = (cat['mu_ave_g'] > 22.5) & (cat['mu_ave_g'] < 29.0)
        ell_cut = cat['ellipticity'] < 0.75
        cat = cat[mu_cut & ell_cut]

        if morph_cut:
            logger.info('applying morphology cuts')
            cat = cat[cat['acorr_ratio'] < 2.5]
        else:
            logger.warn('not applying morphology cuts; is this what you want?')

        cat.reset_index(drop=True, inplace=True)

        if nsa_cut:
            logger.info('removing sources near galaxies with M > {:.2e} Msun'.\
                        format(nsa_min_mass))
            nsa = load_nsa(min_mass=nsa_min_mass)
            logger.info('radius threshold set to {} x r_eff'.\
                        format(nsa_rad_factor))
            nsa_rad = nsa_rad_factor * nsa['sersic_th50'].data
            nsa_coord = SkyCoord(nsa['ra'], nsa['dec'], unit='deg')

            cat_coord = SkyCoord(cat['ra'], cat['dec'], unit='deg')
            nsa_xyz = np.asarray(ra_dec_to_xyz(nsa['ra'], nsa['dec'])).T
            cat_xyz = np.asarray(ra_dec_to_xyz(cat['ra'], cat['dec'])).T
            kdt = KDTree(cat_xyz)
            idx = kdt.query_radius(
                    nsa_xyz, angular_dist_to_euclidean_dist(nsa_rad / 3600.0), 
                    count_only=False, return_distance=False)
            cat_idx = np.unique(
                np.concatenate([_i for _i in idx if len(_i)>0]))

            logger.info('{} sources cut from neighbor cut'.\
                        format(len(cat_idx)))

            cat.drop(cat_idx, inplace=True)
            cat.reset_index(drop=True, inplace=True)
        else:
            logger.warn('not applying nsa galaxy cut; is this what you want?')

        if cirrus_cut:
            logger.info('applying cirrus cut with N_max = {} and Ag_max = {}'.\
                         format(max_sources, max_A_g))

            num_patches_0 = len(cat.groupby(['tract', 'patch'])['id'].count())
            logger.info('{} patches before cirrus cut'.format(num_patches_0))

            cut_func = lambda _x: (_x['A_g'].median() >= max_A_g) |\
                                  (_x['id'].count() >= max_sources)      

            cirrus_patches = cat.groupby(['tract', 'patch']).\
                                         filter(cut_func).\
                                         groupby(['tract', 'patch'])

            cirrus_patches = cirrus_patches[['ra', 'dec', 'A_g']].agg('median')
            cat = cat.set_index(['tract', 'patch']).drop(cirrus_patches.index)
            cat.reset_index(inplace=True)

            num_patches = len(cat.groupby(['tract', 'patch'])['id'].count())

            logger.info('removed {} patches with cirrus cut'.\
                format(num_patches_0 - num_patches))

        else:
            logger.warn('not applying cirrus cut; is this what you want?')
            num_patches = len(cat.groupby(['tract', 'patch'])['id'].count())
            cirrus_patches = None

        logger.info('searched {} patches'.format(num_patches))
        logger.info('{} sources in catalog after cuts'.format(len(cat)))

    return cat, session, engine, cirrus_patches


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
    default_data_path = '/tigress/jgreco/hsc-s18a/hugs-run'

    parser = ArgumentParser()
    parser.add_argument('--run-name', dest='run_name', required=True)
    parser.add_argument('-o', '--outfile', required=True)
    parser.add_argument('--data-path', dest='data_path',
                        default=default_data_path)
    parser.add_argument('--no-cuts', dest='no_cuts', action='store_true') 
    parser.add_argument('--morph-cut', dest='morph_cut', action='store_true') 
    parser.add_argument('--nsa-cut', dest='nsa_cut', action='store_true') 
    parser.add_argument('--nsa-min-mass', default=1e10, type=float)
    parser.add_argument('--nsa-rad-frac', default=5, type=int)
    parser.add_argument('--keep-duplicates', dest='keep_duplicates', 
                        action='store_true') 
    parser.add_argument('--max-sep', dest='max_sep', default=0.2, type=float, 
                        help='count sources as duplicates if they are'
                             'separated by more than this angle in arcsec.')
    parser.add_argument('--random-subsample', default=None, type=int, 
                        dest='random_subsample')
    parser.add_argument('--xmatch-old-cat', dest='xmatch_old_cat', 
                        action='store_true') 
    parser.add_argument('--viz-inspect-cat', dest='viz_cat', 
                        action='store_true')
    parser.add_argument('--no-extinction', dest='no_ext', 
                        action='store_true')
    args = parser.parse_args()

    db_fn = os.path.join(args.data_path, args.run_name)
    db_fn = glob.glob(db_fn + '/*.db')[0]

    logger.info('using database ' + db_fn)
    hugs_cat, session, engine = get_catalog(db_fn, 
                                            args.no_cuts, 
                                            args.morph_cut, 
                                            args.no_ext, 
                                            args.nsa_cut, 
                                            args.nsa_min_mass, 
                                            args.nsa_rad_frac)

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
    
    if args.viz_cat:
        _cols = ['ra', 
                 'dec',  
                 'a_image',
                 'b_image',
                 'theta_image',
                 'ellipticity',
                 'mag_auto_g',
                 'mag_auto_r',
                 'mag_auto_i',
                 'flux_radius_ave_g',
                 'flux_radius_ave_i',
                 'mu_ave_g',
                 'mu_ave_i',
                 'acorr_ratio',
                 'g-i', 
                 'g-r', 
                 'A_g', 
                 'A_r', 
                 'A_i']
        hugs_cat = hugs_cat[_cols]
        hugs_cat['viz-id'] = np.arange(1, len(hugs_cat) + 1)

    hugs_cat.write(args.outfile, overwrite=True)
