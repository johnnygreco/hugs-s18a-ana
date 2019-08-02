import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import gammaincinv, gamma
from astropy.table import Table, vstack
from utils import param_dict, labels
plt.style.use('../notebooks/jpg.mplstyle')


def completeness_grid(injected, recovered, measured, x_par, y_par, 
                      annot_type='none', percent=True, mask_zeros=True,
                      cmap='Purples', label_fs=20, bins=None, dbin=[1,1], 
                      line_color='tab:red', x_bin_pad=[1, 3], y_bin_pad=[1, 3], 
                      frac_masked=0, return_hist_ax=False, xlim=None, 
                      ylim=None, fig_dir=None, fig_label=None):
    
    print('making completeness grid')
    
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

    axHistx.set(xticklabels=[], xticks=ax.get_xticks(), xlim=ax.get_xlim())
                #ylim=(0, 10000))
    axHistx.set_ylabel('Number')
    axHisty.set(yticklabels=[], yticks=ax.get_yticks() ,ylim=ax.get_ylim()) 
                #xlim=(0, 16000))
    axHisty.set_xlabel('Number')
    axHistx.minorticks_on()
    axHisty.minorticks_on()
    axHistx.legend(loc=(1.03, 0.2), fontsize=17)

    fig_label = '' if fig_label is None else fig_label + '-' 
    fn = os.path.join(fig_dir, fig_label + 'completeness-re-muave.png')
    fig.savefig(fn, dpi=200)

    print('making 1d completeness')

    mu_vals = y_centers[1:-1]
    colors = plt.cm.magma(np.linspace(0, 1, 9))

    fig1d, ax1d = plt.subplots(figsize=(8.5, 5.5))
    normalize = mcolors.Normalize(vmin=3, vmax=10)
    colormap = plt.cm.magma

    for i in range(1, 9):
        color = colormap(normalize(x_centers[i]))
        ax1d.plot(mu_vals, H_frac[i, 1:-1], c=color, lw=2.5);

    ax1d.minorticks_on()
    ax1d.set_ylim(0, 1)

    n = 1.0
    b_n = gammaincinv(2. * n, 0.5)
    f_n = gamma(2*n)*n*np.exp(b_n)/b_n**(2*n)
    mu_0 = mu_vals + 2.5*np.log10(f_n) - 2.5*b_n/np.log(10)

    ax1d_2 = ax1d.twiny()
    vals = [0] * len(mu_vals)
    ax1d_2.plot(mu_0, vals, c='w')
    ax1d_2.minorticks_on()

    fs = 22.5

    ax1d_2.set_xlabel(labels['mu_0_g_n1'], fontsize=fs, labelpad=10)
    ax1d.set_xlabel(labels['mu_e_ave_g'], fontsize=fs)
    ax1d.set_ylabel('Detection Efficiency', fontsize=fs)

    # Colorbar setup
    colorparams = x_centers[1:9]
    s_map = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams[1] - colorparams[0])/2.0
    boundaries = np.linspace(colorparams[0] - halfdist, 
                             colorparams[-1] + halfdist, len(colorparams) + 1)

    # Use this to emphasize the discrete color values
    cbar = fig1d.colorbar(s_map, spacing='proportional', 
                          ticks=colorparams, boundaries=boundaries, 
                          format='%2.2g') # format='%2i' for integer

    cbar.ax.tick_params(length=5)
    cbar.ax.set_ylabel(labels['r_e'], fontsize=fs)

    fn = os.path.join(fig_dir, fig_label + 'completeness-1d.png')
    fig1d.savefig(fn, dpi=200)

    return (fig, ax, axHistx, axHisty) if return_hist_ax else (fig, ax)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--color', required=True, 
                        help='gal colors: all, reds, blues, med')
    args = parser.parse_args()

    data_path = '/Users/jgreco/local-io/hugs-data/synth-results'
    fig_dir = '/Users/jgreco/local-io/figures/hugs-ana'

    num_runs = 3

    if args.color == 'all':
        galcolors = ['blues', 'med', 'reds']
        fig_label = 'all-synths'
        cmap = 'Purples'
    elif args.color == 'reds':
        galcolors = ['reds']
        fig_label = 'red-synths'
        cmap = 'Reds'
    elif args.color == 'blues':
        galcolors = ['blues']
        fig_label = 'blue-synths'
        cmap = 'Blues'
    elif args.color == 'med':
        galcolors = ['med']
        fig_label = 'med-synths'
        cmap = 'Greens'
    else:
        raise Exception('not a valid color (reds, blues, med, all)')

    # combine synth results
    synth_cat = []
    synth_match = []
    hugs_match = []

    data = lambda fn: Table.read(os.path.join(data_path, fn))
    
    for color in galcolors:
        for i in range(1, num_runs + 1):
            label = '-{}-0{}.csv'.format(color, i)
            synth_cat.append(data('synth-cat' + label))
            synth_match.append(data('synth-match' + label))
            hugs_match.append(data('hugs-match' + label))
    
    synth_cat = vstack(synth_cat)
    synth_match = vstack(synth_match)
    hugs_match = vstack(hugs_match)

    print(len(synth_cat), 'objects injected')
    print(len(hugs_match), 'objects in hugs match')


    completeness_grid(synth_cat, synth_match, hugs_match, 'r_e', 
                      'mu_e_ave_g', dbin=[1.0, 0.5], x_bin_pad=[1, 3], 
                      y_bin_pad=[0, 3], fig_dir=fig_dir, cmap=cmap,
                      fig_label=fig_label)

    plt.show()
