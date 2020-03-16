"""summary tools for ridge regression

- contains code specific to fMRI datasets
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy

import sys
sys.path.append('/auto/k1/storm/python_path/storm')
from storm.utils import plots as plot_utils


def cv_report(results, model_name=None, subject=None, transform=None,
              ridges=None,
              cmap='viridis', save_dir=None,
              performance_histogram=True,
              global_vs_local=False,
              ridge_histogram=True,
              performance_flatmap=True,
              ridge_flatmap=True,
              file_label=''):
    """create diagnostic plots for results of cv_ridge()

    - specify subject and transform parameters to create flatmaps
    - specify save_dir to save figures to a directory

    Parameters
    ----------
    - results: dict of results from cv_ridge()
    - model_name: str of model name, added to titles of plots
    - subject: str subject surface name
    - transform: str transform name
    - performance_histogram: bool of whether to plot histogram of voxel performance
    - global_vs_local: bool of whether to plot 2d histogram of global vs local
    - ridge_histogram: bool of whether to plot histogram
    """
    if model_name is None:
        model_name = ''
    else:
        model_name += ' '

    if save_dir is not None:
        outputs = {
            'performance_histogram': '{file_label}performance_histogram.png',
            'global_vs_local': '{file_label}global_vs_local.png',
            'ridge_histogram': '{file_label}ridge_histogram.png',
            'performance_flatmap': '{file_label}performance_flatmap.png',
            'ridge_flatmap': '{file_label}ridge_flatmap.png',
        }
        outputs = {
            name: os.path.join(save_dir, path)
            for name, path in outputs.items()
        }
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # test set performance plots
    if performance_histogram:
        histograms = {}
        if 'global_performance' in results:
            histograms['global'] = results['global_performance']
        if 'local_performance' in results:
            histograms['local'] = results['local_performance']
        plot_utils.hists1d(
            histograms,
            title=(model_name + 'test set performance'),
        )
        if save_dir is not None:
            plt.savefig(outputs['performance_histogram'])
        plt.show()

    if global_vs_local:
        plot_utils.hist2d(
            results['global_performance'],
            results['local_performance'],
            xlabel='global',
            ylabel='local',
            clabel='# regressands',
            title=(model_name + 'global vs local test set performance'),
        )
        if save_dir is not None:
            plt.savefig(outputs['global_vs_local'])
        plt.show()

    # ridge parameter histogram plots
    if ridge_histogram:
        bin_counts, (x_bin_bounds, y_bin_bounds) = plot_utils.hist1d(
            results['local_optimal_ridge'],
            stacked_data=results['local_performance'],
            log_x=True,
            n_bins=len(ridges),
            n_stacked_bins=10,
            stacked_cmap='spectral',
            xlabel='ridge parameter',
            ylabel='regressands',
            title=(model_name + 'ridge parameters'),
            stacked_label='test performance',
        )
        global_optimum = results['global_optimal_ridge']
        global_index = np.nonzero(x_bin_bounds <= global_optimum)[0][-1]
        bounds = x_bin_bounds
        plt.plot(
            scipy.stats.gmean([bounds[global_index], bounds[global_index + 1]]),
            bin_counts.sum(1)[global_index],
            '.k',
            marker='*',
            markersize=15,
            label='global optimum',
        )
        plt.legend(numpoints=1, fontsize=20)
        if save_dir is not None:
            plt.savefig(outputs['ridge_histogram'])
        plt.show()

    # fMRI-specific reports
    if subject is not None and transform is not None:
        import cortex

        # performance flatmap
        if performance_flatmap:
            volume = cortex.Volume(
                results['local_performance'],
                subject=subject,
                xfmname=transform,
                cmap=cmap,
                vmin=0,
                vmax=.5,
                colorbar_ticklabelsize=15,
            )
            cortex.quickshow(volume)
            plt.title(model_name + 'test performance', fontsize=30)
            if save_dir is not None:
                plt.savefig(outputs['performance_flatmap'])
            plt.show()

        # ridge parameter flatmap
        if ridge_flatmap:
            volume = cortex.Volume(
                -np.log(results['local_optimal_ridge']),
                subject=subject,
                xfmname=transform,
                cmap=cmap,
                vmin=-np.log(ridges[-1]),
                vmax=-np.log(ridges[0]),
            )

            if len(ridges) > 5:
                ticks = np.logspace(
                    np.log10(ridges[0]),
                    np.log10(ridges[-1]),
                    5,
                )
            else:
                ticks = ridges

            cortex.quickshow(
                volume,
                colorbar_ticks=-np.log(ticks),
                colorbar_ticklabels=['{:0.2f}'.format(tick) for tick in ticks],
                colorbar_ticklabelsize=15,
            )
            plt.title(model_name + 'ridge parameters', fontsize=30)
            if save_dir is not None:
                plt.savefig(outputs['ridge_flatmap'])
            plt.show()
