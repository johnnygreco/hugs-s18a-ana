import os
import numpy as np
from astropy.table import Table
from hugs.log import logger

project_dir = os.path.dirname(os.path.dirname(__file__))


param_dict = dict(
    mu_e_ave_g = 'mu_ave_g',
    mu_e_ave_i = 'mu_ave_i',
    m_g = 'mag_auto_g',
    m_r = 'mag_auto_r',
    m_i = 'mag_auto_i',
    r_e = 'flux_radius_ave_g',
    ell='ellipticity'
    
)


labels = dict(
    m_g = r'$m_g$',
    m_r = r'$m_r$',
    m_i = r'$m_i$',
    mu_e_ave_g = r'$\bar{\mu}_\mathrm{eff}(g)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    mu_e_ave_r = r'$\bar{\mu}_\mathrm{eff}(g)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    mu_e_ave_i = r'$\bar{\mu}_\mathrm{eff}(g)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    r_e = r'$r_\mathrm{eff}$ [arcsec]',
    ell = 'Ellipticity'
)


def load_nsa(path='/tigress/jgreco/data/catalogs', min_mass=None):
    """
    Return an astropy table of the NASA Sloan.
    Note: Cannot convert a table with multi-dimensional 
    columns to a pandas DataFrame.
    """
    fn = os.path.join(path, 'nsa_v0_1_2.fits') 
    cat = Table.read(fn)
    for name in cat.colnames:
        cat.rename_column(name, name.lower())
    cat['mass_h07'] = cat['mass']/0.7**2

    if min_mass is not None:
        logger.info('cutting mass at M_star = {:.2e} M_sun'.format(min_mass))
        nsa_host = cat[cat['mass_h07'] > min_mass]
    else:
        nsa_host = cat 

    bad_nsaid = []
    flags = ['bad', 'saturated star', 'bright star',\
             'nearby star', 'satellite trail']

    comment_fn = os.path.join(path, 'nsa-comments.txt')
    with open(comment_fn, 'r') as comment_file:
        lines = comment_file.readlines()

    for l in lines[1:]:
        substr = l[:-2].split('"')[-1]
        nsaid = int(l[:-2].split('"')[0].split()[1])
        for flag in flags:
            if flag in substr.lower():
                bad_nsaid.append(nsaid)

    # most of these are already commented
    nsa_junk = [35263, 1133, 2716, 8808, 21915, 168561, 165191, 27772,
                33424, 61193, 66992, 76794, 105067, 142969, 147262,162751, 
                133245, 26956, 4531, 35513, 50620, 87979, 94798, 131404,
                135031, 146361, 3038, 166644, 136108]
      
    bad_nsaid = np.unique(bad_nsaid + nsa_junk)

    nsa_mask = np.ones_like(nsa_host, dtype=bool)
    for idx, src in enumerate(nsa_host):
        if src['nsaid'] in bad_nsaid:
            nsa_mask[idx] = False
            
    logger.info('will mask {} sources with negative comments'.\
        format((~nsa_mask).sum()))

    nsa_host = nsa_host[nsa_mask]

    # cut from Geha et al. 2018 (SAGA)
    sersic_petro_90_mask = nsa_host['sersic_th50'] < nsa_host['petroth90'] 
    logger.info('will mask {} sources with r_50(sersic) > r_90(petro)'.\
        format((~sersic_petro_90_mask).sum()))
    nsa_host = nsa_host[sersic_petro_90_mask]

    total = (~nsa_mask).sum() + (~sersic_petro_90_mask).sum()
    logger.info('total bad nsa detections removed = {}'.format(total))

    return nsa_host
