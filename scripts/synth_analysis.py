import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from astropy.table import Table
from collections import namedtuple
from astropy.coordinates import SkyCoord
from astropy import units as u

import hugs
from hugs.database.tables import Source, Synth
from hugs.log import logger
from utils import param_dict, labels, project_dir
from build_catalog import get_catalog
plt.style.use(os.path.join(project_dir, 'scripts/jpg.mplstyle'))


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


def completeness_grid(injected, recovered, measured, x_par, y_par, 
                      annot_type='none', percent=True, mask_zeros=True,
                      cmap='Purples', label_fs=20, bins=None, dbin=[1,1], 
                      line_color='tab:red', x_bin_pad=[1, 3], y_bin_pad=[1, 3], 
                      frac_masked=0, return_hist_ax=False, xlim=None, 
                      ylim=None, fig_dir=None, fig_label=None):
    
    logger.info('making completeness grid')

    if bins is None:
        bins = [np.arange(np.floor(injected[x_par].min()) -\
                                    x_bin_pad[0]*dbin[0], 
                          np.ceil(injected[x_par].max()) +\
                                  x_bin_pad[1]*dbin[0], dbin[0]),
                np.arange(np.floor(injected[y_par].min()) -\
                                    y_bin_pad[0]*dbin[1], 
                          np.ceil(injected[y_par].max()) +\
                                  y_bin_pad[1]*dbin[1], dbin[1])]        

    H_injected, _, _ = np.histogram2d(injected[x_par],  
                                      injected[y_par], 
                                      bins=bins)
    H_measured, _, _ = np.histogram2d(recovered[x_par], 
                                      recovered[y_par], 
                                      bins=bins)
    H_injected[H_injected==0] = 1e-8
    H_injected *= (1 - frac_masked)
    H_frac = H_measured/H_injected
    H_frac[H_frac<1e-5] = np.nan

    x_centers = 0.5 * (bins[0][1:] + bins[0][:-1]) - 0.5*dbin[0]
    y_centers = 0.5 * (bins[1][1:] + bins[1][:-1]) - 0.5*dbin[1]
    percent = 100.0 if percent else 1.0    
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_main = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
   
    fig = plt.figure(figsize=(9, 8))

    ax = plt.axes(rect_main)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    mask = (H_frac.T==0) if mask_zeros else None
    
    cax = ax.pcolormesh(x_centers, y_centers, percent * H_frac.T, 
                        cmap=cmap, vmin=0, vmax=percent)
    cbaxes = plt.axes([left, -0.015, width, 0.03]) 
    cbar = fig.colorbar(cax, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Detection Efficiency (\%)')

    ax.set_xlabel(labels[x_par], fontsize=label_fs)
    ax.set_ylabel(ylabel=labels[y_par], fontsize=label_fs)
    ax.minorticks_on()

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    
    weights = np.ones_like(injected[x_par]) * (1 - frac_masked)
    
    axHistx.hist(injected[x_par], bins=x_centers, color='gray', alpha=0.5, 
                 label='Injected', weights=weights)
    
    axHistx.hist(recovered[x_par], bins=x_centers, color='k', alpha=0.5, 
                 label='Recovered')
    axHistx.hist(measured[param_dict[x_par]], bins=x_centers, color=line_color, 
                 label='Measured', alpha=1, lw=3, histtype='step')

    axHisty.hist(injected[y_par], bins=y_centers, color='gray', alpha=0.5,
                 orientation='horizontal', weights=weights)
    axHisty.hist(recovered[y_par], bins=y_centers, color='k', alpha=0.5,
                 orientation='horizontal')
    axHisty.hist(measured[param_dict[y_par]], bins=y_centers, color=line_color, 
                 alpha=1, histtype='step', lw=3, orientation='horizontal')

    axHistx.set(xticklabels=[], xticks=ax.get_xticks(), xlim=ax.get_xlim(), 
                ylim=(0, 16000))
    axHistx.set_ylabel('Number')
    axHisty.set(yticklabels=[], yticks=ax.get_yticks() ,ylim=ax.get_ylim(), 
                xlim=(0, 20000))
    axHisty.set_xlabel('Number')
    axHistx.minorticks_on()
    axHisty.minorticks_on()
    axHistx.legend(loc=(1.03, 0.2), fontsize=17)

    if fig_dir is not None:
        fig_label = '' if fig_label is None else fig_label + '-' 
        fn = os.path.join(fig_dir, fig_label + 'completeness-re-muave.png')
        fig.savefig(fn, dpi=200)

    return (fig, ax, axHistx, axHisty) if return_hist_ax else (fig, ax)


def _clip_outlier_mask(cat, param_name=None, percentiles=[0.001, 99.999]):
    if param_name is not None:
        pmin, pmax = np.percentile(cat[param_name], percentiles)
        mask = (cat[param_name] > pmin) & (cat[param_name] < pmax)
    else: 
        pmin, pmax = np.percentile(cat, percentiles)
        mask = (cat > pmin) & (cat < pmax)
    return mask


def _get_clipped_param(hugs_cat, hugs_param_name, synth_cat, synth_param_name, 
                       **kwargs):
    mask = _clip_outlier_mask(hugs_cat, hugs_param_name, **kwargs)
    hugs_param = hugs_cat[hugs_param_name][mask]
    synth_param = synth_cat[synth_param_name][mask]
    return hugs_param, synth_param


def _get_16_50_84(param):
    lo, med, hi = np.percentile(param, [16, 50, 84])
    return lo, med, hi
    

def parameter_accuracy(hugs_cat, synth_cat, fig_dir, fontsize=22, 
                       fig_label=None, color_mag='mag_ap9'):
 
    logger.info('making parameter accuracy plots')

    f1, a1 = plt.subplots(3, 2, figsize=(10, 10))
    f1.subplots_adjust(hspace=0.35)
    
    hugs_r_e, synth_r_e = _get_clipped_param(
        hugs_cat, 'flux_radius_ave_g', synth_cat, 'r_e', 
        percentiles=[0.1, 99.9])
    
    a1[0, 0].plot(synth_r_e, hugs_r_e, ',', alpha=0.2)
    a1[0, 0].plot([synth_r_e.min(), synth_r_e.max()],
                  [synth_r_e.min(), synth_r_e.max()],
                  'k--', lw=2, zorder=10)
    a1[0, 1].hist(synth_r_e - hugs_r_e, bins='auto', alpha=0.5)
    a1[0, 1].axvline(x=0, ls='-', lw=2, color='k')
    a1[0, 1].set_xlim(-5, 5)
    a1[0, 0].set_xlabel(r'$r_e$', fontsize=fontsize)
    a1[0, 0].set_ylabel('flux radius ave g', fontsize=fontsize-2)
    a1[0, 1].set_xlabel(r'$\delta r_e$', fontsize=fontsize)
    stat_lines = _get_16_50_84(
        synth_cat['r_e'] - hugs_cat['flux_radius_ave_g'])
    for stat in stat_lines:
        a1[0, 1].axvline(x=stat, ls='--', lw=2, c='tab:red', alpha=0.7)
    
    hugs_m_g, synth_m_g = _get_clipped_param(
    hugs_cat, 'mag_auto_g', synth_cat, 'm_g', percentiles=[0.001, 99.999])
    a1[1, 0].plot(synth_m_g, hugs_m_g, ',', alpha=0.2)
    a1[1, 0].plot([synth_m_g.min(), synth_m_g.max()],
                  [synth_m_g.min(), synth_m_g.max()],
                  'k--', lw=2, zorder=10)
    a1[1, 1].hist(synth_m_g - hugs_m_g, bins='auto', alpha=0.5)
    a1[1, 1].axvline(x=0, ls='-', lw=2, color='k')
    a1[1, 1].set_xlim(-1, 1)
    a1[1, 0].set_xlabel(r'$m_g$', fontsize=fontsize)
    a1[1, 0].set_ylabel('mag auto g', fontsize=fontsize-2)
    a1[1, 1].set_xlabel(r'$\delta m_g$', fontsize=fontsize)
    stat_lines = _get_16_50_84(synth_cat['m_g'] - hugs_cat['mag_auto_g'])
    for stat in stat_lines:
        a1[1, 1].axvline(x=stat, ls='--', lw=2, c='tab:red', alpha=0.7)
        
    hugs_mu_ave, synth_mu_ave = _get_clipped_param(
        hugs_match, 'mu_ave_g', synth_cat, 'mu_e_ave_g', 
        percentiles=[0.001, 99.999])
    a1[2, 0].plot(synth_mu_ave, hugs_mu_ave, ',', alpha=0.2)
    a1[2, 0].plot([synth_mu_ave.min(), synth_mu_ave.max()],
                  [synth_mu_ave.min(), synth_mu_ave.max()],
                  'k--', lw=2, zorder=10)
    a1[2, 0].set_ylim(22, 30)
    a1[2, 1].hist(synth_mu_ave - hugs_mu_ave, bins='auto', alpha=0.5)
    a1[2, 1].axvline(x=0, ls='-', lw=2, color='k')
    a1[2, 1].set_xlim(-1, 1)
    a1[2, 0].set_xlabel(r'$\bar{\mu}_\mathrm{eff}(g)$', fontsize=fontsize)
    a1[2, 0].set_ylabel('mu ave g', fontsize=fontsize-2)
    a1[2, 1].set_xlabel(r'$\delta \bar{\mu}_\mathrm{eff}(g)$', 
                        fontsize=fontsize)
    stat_lines = _get_16_50_84(
        synth_cat['mu_e_ave_g'] - hugs_match['mu_ave_g'])
    for stat in stat_lines:
        a1[2, 1].axvline(x=stat, ls='--', lw=2, c='tab:red', alpha=0.7)

    synth_gi = synth_match['g-i'][0]
    synth_gr = synth_match['g-r'][0]

    hugs_gi = hugs_match[color_mag + '_g'] - hugs_match[color_mag + '_i']
    hugs_gr = hugs_match[color_mag + '_g'] - hugs_match[color_mag + '_r']
    
    hugs_gi = hugs_gi[_clip_outlier_mask(hugs_gi, percentiles=[0.1, 99.9])]
    hugs_gr = hugs_gr[_clip_outlier_mask(hugs_gr, percentiles=[0.1, 99.9])]
    
    f2, a2 = plt.subplots()
    a2.hist(hugs_gi, bins='auto', label=r'$g-i$', alpha=0.5)
    a2.axvline(x=synth_gi, ls='--', lw=2, color='k')
    a2.hist(hugs_gr, bins='auto', label=r'$g-r$', alpha=0.5)
    a2.axvline(x=synth_gr, ls='--', lw=2, color='k')
    a2.legend(loc=0, fontsize=18)
    a2.set_xlabel('color', fontsize=fontsize)

    fig_label = '' if fig_label is None else fig_label + '-' 
    fn_1 = os.path.join(fig_dir, fig_label + 'accuracy-r-mag-muave.png')
    f1.savefig(fn_1, dpi=200)

    fn_2 = os.path.join(fig_dir, fig_label + 'accuracy-color.png')
    f2.savefig(fn_2, dpi=200)


def get_injected_synth_ids(db_fn=None, session=None, engine=None):

    if db_fn is not None:
        logger.info('connecting to hugs database')
        engine = hugs.database.connect(db_fn)
        session = hugs.database.Session()
    else:
        assert session is not None and engine is not None

    query = session.query(Synth)
    synth_ids = pd.read_sql(query.statement, engine)

    return synth_ids


def _get_unique_synths(synth_ids):
    synth_id_unique, synth_id_idx = np.unique(synth_ids['synth_id'].values, 
                                              return_index=True)
    injected_synth = synth_ids.iloc[synth_id_idx]    

    return synth_id_unique, injected_synth


if __name__ == '__main__':
    
    default_synth_dir = '/tigress/jgreco/hsc-s18a/synths/global'
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-name', dest='run_name')
    group.add_argument('--cat-fn', dest='cat_fn', default=None)
    parser.add_argument('--synth-dir', dest='synth_dir', 
                        default=default_synth_dir)
    parser.add_argument(
        '--synth-cat-fn', dest='synth_cat_fn', 
        default=os.path.join(default_synth_dir, 'global-synth-cat.fits'))
    parser.add_argument('--fig-dir', dest='fig_dir', 
                        default=os.path.join(project_dir, 'figs'))
    parser.add_argument('--no-cuts', dest='no_cuts', action='store_true')
    parser.add_argument('--save-fn', dest='save_fn', default=None)
    parser.add_argument('--min-sep', dest='min_sep', default=1.0, type=float)
    parser.add_argument('--fig-label', dest='fig_label', default=None)
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

    logger.info('{:.1f}% of synths injected'.\
                format(100 * len(synth_id_unique) / len(synth_cat)))

    masked = injected_synth['mask_bright_object'].values.astype(bool)
    masked |= injected_synth['mask_no_data'].values.astype(bool)
    masked |= injected_synth['mask_sat'].values.astype(bool)
    masked_frac = masked.sum()/len(injected_synth)
    logger.info('{:.1f}% of synths masked by HSC'.format(100 * masked_frac))

    masked_frac = injected_synth['mask_small'].sum() / len(injected_synth)
    logger.info('{:.1f}% of synths masked: SMALL'.format(100 * masked_frac))

    masked_frac = injected_synth['mask_cleaned'].sum() / len(injected_synth)
    logger.info('{:.1f}% of synths masked: CLEANED'.format(100 * masked_frac))

    synth_cat = synth_cat[synth_id_unique - 1]
    synth_cat = synth_cat[~masked]

    hugs_match, synth_match = match_synths(
        hugs_cat, synth_cat, min_sep=args.min_sep*u.arcsec)

    logger.info('recovered {:.1f}% of the synths'.\
                format(100 * len(synth_match) / len(synth_cat)))
    
    logger.info('found synths / all sources ~ {:.1f}%'.\
                format(100 * len(synth_match) / len(hugs_cat)))

    logger.info('# of sources in catalog = {}'.format(len(hugs_cat)))
    logger.info('# of sources synths in catalog = {}'.format(len(synth_match)))

    parameter_accuracy(hugs_match, synth_match, args.fig_dir, 
                       fig_label=args.fig_label)

    completeness_grid(synth_cat, synth_match, hugs_match, 'r_e', 'mu_e_ave_g', 
                      dbin=[0.5, 0.5], x_bin_pad=[1, 3], fig_dir=args.fig_dir, 
                      fig_label=args.fig_label)
