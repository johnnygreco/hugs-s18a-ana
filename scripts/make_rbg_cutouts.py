from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from  multiprocessing import Pool
from functools import partial
from astropy.table import Table
from astropy import units as u
plt.style.use('jpg.mplstyle')

import lsst.log
Log = lsst.log.Log()
Log.setLevel(lsst.log.ERROR)

import lsst.daf.persistence
import lsstutils
from hugs.utils import mkdir_if_needed, angsep
ROOT = '/tigress/HSC/DR/s18a_wide'


def _get_skymap(root=ROOT):
    butler = lsst.daf.persistence.Butler(root)
    skymap = butler.get('deepCoadd_skyMap', immediate=True)
    print()
    return butler, skymap


def _draw_ellipse(ra, dec, ell_pars, wcs, ax, color='c', **kwargs):
    r_e, theta, ell, scale = ell_pars
    r_e_pix = r_e/0.168
    q = 1.0 - ell
    diam = 2*r_e_pix
    x, y = wcs.skyToPixel(lsstutils.make_afw_coords([ra, dec]))
    e = Ellipse([x, y], scale*diam, scale*diam*q, angle=theta,  
                ec=color, fc='none', lw=1.5, ls='--', **kwargs)
    ax.add_patch(e)


def single_rgb_image(ra, dec, radius, prefix, Q=8., dataRange=0.6, scale=20, 
                     file_format='png', img_size=None, butler=None, 
                     skymap=None, root=ROOT, dpi=150, ell_pars=None, 
                     full_cat=None, ell_scale=1):

    if butler is None:
        butler, skymap = _get_skymap(root)

    try:
        img, wcs = lsstutils.make_rgb_image(
            ra, dec, radius, Q=Q, dataRange=dataRange, 
            butler=butler, skymap=skymap, img_size=img_size, 
            return_wcs=True)
    except: 
        print('WARNING: failed to get {} at {}, {}'.format(prefix, ra, dec))
        return None

    if img is not None:

        fig, ax = plt.subplots(
            subplot_kw={'yticks':[], 'xticks':[]})
        ax.imshow(img, origin='lower')

        if scale:
            shape = img.shape
            xmin = 15.0
            xmax = xmin + scale/0.168
            y=0.93*shape[0]
            ax.axhline(y=y, xmin=xmin/shape[1], xmax=xmax/shape[1], 
                       color='w', lw=3.0, zorder=1000)
            label = str(int(scale))
            ax.text((xmin+xmax)/2 - 0.042*shape[1], y - 0.072*shape[0], 
                    r'$'+label+'^{\prime\prime}$', color='w', fontsize=20)

        if ell_pars is not None:
            _draw_ellipse(ra, dec, ell_pars, wcs, ax, alpha=0.8, zorder=100)

        if full_cat is not None:
            ra_c, dec_c = wcs.getSkyOrigin()
            ra_c = ra_c.asDegrees()
            dec_c = dec_c.asDegrees()
            seps = angsep(ra_c, dec_c, full_cat['ra'], full_cat['dec']) 
            src_cut= seps < 2 * radius
            sources = full_cat[src_cut]
            for src in sources:
                ell_pars = src['flux_radius_ave_g'], src['theta_image'],\
                           src['ellipticity'], ell_scale
                _draw_ellipse(src['ra'], src['dec'], ell_pars, wcs, 
                              ax, color='lightgray', alpha=0.7, zorder=10)

        fig.savefig(prefix+'.'+file_format, bbox_inches='tight', 
                    pad_inches=0, dpi=dpi)

        plt.close('all')


def _mp_run(obj, extra_args):
    out_path, ellipse_scale, radius = extra_args[:3]
    Q, dataRange, scale, file_format, img_size = extra_args[3:8]
    skymap, dpi, full_cat, butler = extra_args[8:]

    num = obj['viz-id']

    print('source:', num)
    new_prefix = os.path.join(out_path, 'hugs-'+str(num))
    if ellipse_scale is not None:
        ell_pars = obj['flux_radius_ave_g'], obj['theta_image'],\
                   obj['ellipticity'], ellipse_scale
    else:
        ell_pars = None
    single_rgb_image(
        obj['ra'], obj['dec'], radius, new_prefix, Q, dataRange, scale,
        file_format, img_size, butler=butler, skymap=skymap, dpi=dpi,
        ell_pars=ell_pars, ell_scale=ellipse_scale, full_cat=full_cat)


def batch_rgb_images(cat_fn, radius, out_path, Q=8, dataRange=0.6, scale=20,
                     file_format='png', img_size=None, root=ROOT, dpi=150,
                     ellipse_scale=None, full_cat=None, nproc=1):

    cat = Table.read(cat_fn)
    butler, skymap = _get_skymap(root)

    mkdir_if_needed(out_path)

    print('generating {} rgb images for...'.format(len(cat)))

    if nproc == 1:
        for obj in cat:
            num = obj['viz-id']
            print('source:', num)
            new_prefix = os.path.join(out_path, 'hugs-'+str(num))
            if ellipse_scale is not None:
                ell_pars = obj['flux_radius_ave_g'], obj['theta_image'],\
                           obj['ellipticity'], ellipse_scale
            else:
                ell_pars = None
            single_rgb_image(
                obj['ra'], obj['dec'], radius, new_prefix, Q, dataRange, scale, 
                file_format, img_size, butler=butler, skymap=skymap, dpi=dpi,
                ell_pars=ell_pars, ell_scale=ellipse_scale, full_cat=full_cat)
    else:
        extra_args = [
            out_path, ellipse_scale, radius, Q, dataRange, scale, 
            file_format, img_size, skymap, dpi, full_cat, butler
        ]

        pool_func = partial(_mp_run, extra_args=extra_args)

        with Pool(nproc) as pool:
            pool.map(pool_func, cat)


if __name__=='__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-s', '--single', type=float, nargs=2, default=None,
        help='single mode: ra dec (in deg)')
    group.add_argument(
        '-b', '--batch-fn', dest='batch_fn', type=str, default=None,
        help='batch mode: csv/fits catalog (with ra & dec) file name')
    parser.add_argument(
        '-o', '--output', type=str, required=True, 
        help='output file (single mode) prefix or directory (batch mode).') 
    parser.add_argument(
        '--full-cat-fn', dest='full_cat_fn', type=str, default=None, 
        help='catalog of all sources to plot in addition to main source') 
    parser.add_argument(
        '-r', '--radius', type=float, default=35,
        help='angular radius of cutout in arcsec')
    parser.add_argument(
        '-Q', type=float, default=8, help='RGB function parameter')
    parser.add_argument(
        '--dataRange', type=float, default=0.6, help='RGB function parameter')
    parser.add_argument(
        '--format', type=str, default='png', help='file format')
    parser.add_argument(
        '--root', type=str, default=ROOT,
        help='Root data directory.')
    parser.add_argument('--dpi', type=float, default=150)
    parser.add_argument(
        '--ell-scale', dest='ell_scale', type=float, default=None,
        help='draw an ellipse on image scaled by this value (batch mode only)')
    parser.add_argument('--nproc', type=int, default=1)

    args = parser.parse_args()

    if args.full_cat_fn is not None:
        full_cat = Table.read(args.full_cat_fn)
    else:
        full_cat = None

    if args.single:
        ra, dec, = args.single
        single_rgb_image(
            ra, dec, args.radius, args.output, args.Q, 
            args.dataRange, file_format=args.format, root=args.root, 
            dpi=args.dpi, full_cat=full_cat)
    elif args.batch_fn:
        batch_rgb_images(
            args.batch_fn, args.radius, args.output, args.Q, 
            args.dataRange, file_format=args.format, root=args.root, 
            dpi=args.dpi, ellipse_scale=args.ell_scale, full_cat=full_cat, 
            nproc=args.nproc)
