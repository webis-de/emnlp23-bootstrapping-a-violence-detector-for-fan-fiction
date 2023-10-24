from pathlib import Path

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import f1_score


def create_plot(fig, clf_types, result_files, subset_name, show_y_label=True, show_legend=True):

    ax = fig.subplots(8,10)
    hist_ax = plt.subplot2grid((10, 10), (0, 0), colspan=10, rowspan=3, fig=fig)
    result_ax = plt.subplot2grid((10, 10), (3, 0), colspan=10, rowspan=7, fig=fig)

    bins = [0, 1000, 10_000, 100_000, 1_000_000]

    # read first file to plot the document length histogram
    df = pd.read_csv(result_files[0], sep='\t')
    bin_hist = df['content_num_tokens'].value_counts(
        ascending=True, 
        sort=False, 
        bins=bins
    ).values

    x = np.array(['<1k', '1-10k', '10-100k', '100k-1M'])
    sns.barplot(x=x, y=bin_hist, ax=hist_ax, color='#336699')
    hist_ax.set_xticklabels([])
    if show_y_label:
        hist_ax.set_ylabel('documents')
    hist_ax.set_ylim([0, np.max(bin_hist) + 200])
    hist_ax.set_title(subset_name)

    # plot results for each clf_type
    pal = sns.color_palette('tab10')
    num_bins = len(bins) - 1
    bin_labels = np.arange(num_bins)


    for i, (clf_type, result_file) in enumerate(zip(clf_types, result_files)):
        df = pd.read_csv(result_file, sep='\t')
        binning_result = pd.cut(df['content_num_tokens'],
                                bins=bins,
                                labels=bin_labels,
                                right=False)
        
        binned_score = np.zeros(num_bins)
        for bin_label in bin_labels:
            df_bin = df[binning_result == bin_label]
            if df_bin.empty:
                binned_score[bin_label] = np.nan
                continue
            binned_score[bin_label] = f1_score(df_bin['target_label'], df_bin['prediction'])
        
        binned_score = binned_score.astype(float)
        with plt.rc_context({'lines.linestyle': '--'}):
            ax_sub = sns.pointplot(x=x, y=binned_score, ax=result_ax, color=pal.as_hex()[i],
                                linestyles='dotted', label=clf_type)
    
    result_ax.set_ylim([0.0, 1.0])
    if show_y_label:
        result_ax.set_ylabel('$F_1$ Score')

    if show_legend:
        handles = [
            mpatches.Patch(color=pal.as_hex()[i], label=clf_type)
            for i, clf_type in enumerate(clf_types)
        ]
        result_ax.legend(handles=handles)
    
    plt.subplots_adjust(wspace=1, hspace=1)


def create_plots(data, plt_file):

    subset_names = list(data.keys())
    width, height = 6, 4
    fig = plt.figure(figsize=(width * len(subset_names), height))
    # take first row, as only one row is created!
    sub_figs = fig.subfigures(1, len(subset_names), squeeze=False)[0]
    
    for i, (subset_name, fig) in enumerate(zip(subset_names, sub_figs)):
        clf_types = [x[0] for x in data[subset_name]]
        result_files = [x[1] for x in data[subset_name]]
        is_first = (i == 0)
        create_plot(fig, clf_types, result_files, subset_name, show_y_label=is_first, show_legend=is_first)

    plt.savefig(plt_file, dpi=600)


if __name__ == '__main__':
    
    d = Path('/mnt/ceph/storage/data-tmp/current/cschroeder/trigger-warning-classification')
    data = {
        'fame-balanced': [
            ("bert", d / 'predictions_bert_fame-balanced-test.tsv'),
            ("svm", d / 'predictions_svm_fame-balanced-test.tsv'),
        ],
        'random-balanced': [
            ("bert", d / 'predictions_bert_random-balanced-test.tsv'),
            ("svm", d / 'predictions_svm_random-balanced-test.tsv'),
        ],
        'tag-frequency-balanced': [
            ("bert", d / 'predictions_bert_tag-frequency-balanced-test.tsv'),
            ("svm", d / 'predictions_svm_tag-frequency-balanced-test.tsv'),
        ]
    }
    plt_file = Path('/mnt/ceph/storage/data-tmp/current/ob14dupe/trigger-warning-classification/plt.svg')
    create_plots(data, plt_file)
