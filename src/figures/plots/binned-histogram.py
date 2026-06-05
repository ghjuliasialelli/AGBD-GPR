"""

This script plots the boxplots of the binned histograms for pre- and post-kriging residuals.
It creates the file `boxplot_residuals.png` with the comparison.

"""


#######################################################################################################################
# Imports

from config import DATA_ROOT
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
from os import makedirs

all_colors = [
    '#EE6677',  # rose (I)
    '#228833',  # green (II)
    '#CCBB44',  # yellow (III)
    '#66CCEE',  # cyan (IV)
    '#AA3377',  # purple (V)
    '#BBBBBB',  # grey (VI)
    '#EE8866',  # orange (VII)
    '#44BB99',  # teal (VIII)
    '#332288',  # indigo (IX)
]

""" old, pre-rebuttal 
name_to_id = {
    '46171204': "I",
    '46617715': "II",
    '46637671': "III",
    '47118870': "IV",
    '47198254': "V",
    '47250199': "VI",
    '48428326': "VII",
    '50185245': "VII (10K)",
    '50140225' : "VII (5K)",
    '50141503' : "VII*",
    '1617922' : 'VIII',
    'CCI': "ESA CCI"
}
"""

name_to_id = {
    '46171204': "I",
    '46617715': "II",
    '46637671': "III",
    '1757019': "IV",
    '1617922': "V",
    '1790043': "VI",
    '1820465': "VII",
    '47250199': "VIII",
    '48428326': "IX",
    '50185245': "IX (10K)",
    '50140225' : "IX (5K)",
    '50141503' : "IX*",
    'CCI': "ESA CCI"
}

#######################################################################################################################
# Helper functions

def box_stats_from_hist(counts, bin_edges):
    """
    This function computes boxplot statistics from a histogram.

    Args:
    - counts (np.ndarray): Counts of the histogram bins.
    - bin_edges (np.ndarray): Edges of the histogram bins.

    Returns:
    - stats (dict): Dictionary containing boxplot statistics (q1, median, q3, whisker_low, whisker_high).
    """
    total = counts.sum()
    if total == 0:
        return None
    cdf = np.cumsum(counts) / total
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    def percentile(p): return centers[np.searchsorted(cdf, p/100.0)]
    q1 = percentile(25)
    median = percentile(50)
    q3 = percentile(75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    # whiskers are the most extreme *data* points inside the fences
    valid = centers[(centers >= lower_fence) & (centers <= upper_fence) & (counts>0)]
    whisker_low = valid.min() if len(valid) else q1
    whisker_high = valid.max() if len(valid) else q3
    stats = {
        'q1': q1,
        'median': median,
        'q3': q3,
        'whisker_low': whisker_low,
        'whisker_high': whisker_high,
        'total_count': total
    }
    return stats

def numeric_key(k):
    # extract the first number before the dash and convert to int
    return int(re.match(r"(\d+)", k).group(1))

def draw_box(ax, pos, stats, color, width):
    """Draw a single box/whisker at horizontal position pos."""
    q1, med, q3 = stats['q1'], stats['median'], stats['q3']
    wlo, whi = stats['whisker_low'], stats['whisker_high']
    # box
    ax.add_patch(Rectangle((pos - width/2, q1),
                        width, q3 - q1,
                        facecolor=color, edgecolor='black', alpha=0.7))
    # median line
    ax.hlines(med, pos - width/2, pos + width/2, color='black', lw=1)
    # whiskers
    ax.vlines(pos, wlo, whi, color='black', lw=0.5)
    ax.hlines([wlo, whi], pos - width/4, pos + width/4, color='black', lw=0.5)


def get_colors(color, N) :
    cmap = plt.get_cmap(color)
    colors = [cmap(i) for i in np.linspace(0.3, 1, N)]  # 0.3–1 avoids very pale
    return colors


def compare_min_bins(models, test, subset, min_values, _max, regional = False) :
    """
    This function compares boxplots for different minimum bin values.

    Args:
    - models (list): List of model names to compare.
    - test (bool): Whether to use test set pre-kriging histograms.
    - subset (bool): Whether to use subsetted pre-kriging histograms.
    - min_values (list): List of minimum bin values to compare.
    - _max (int): Maximum bin value.
    - regional (bool): Whether to plot regional histograms.

    Returns:
    - None: Saves the boxplot figure.
    
    Execute as, e.g.: compare_min_bins(['46783899'], test, subset, [500,400], _max, regional)
    """

    region = 'global' 
    res = {}

    # Assign colors
    pre_colors = get_colors('Blues', len(min_values))
    post_colors = get_colors('Reds', len(models) * len(min_values))
    colors_dict = {}

    for i, _min in enumerate(min_values) :

        suffix = f"{'_subset' if subset else ''}{'_test' if test else ''}{'_regional' if regional else ''}{('_' + str(_min) + '-' + str(_max)) if _min != 400 else ''}"
        residual_bins = np.arange(-_min, _max, 1)

        # Load pre-kriging binned histograms
        if test :
            with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/hist-pre{suffix}_test-{models[0]}.pkl', 'rb') as f:
                if regional : pre_bin_hists = pickle.load(f)['binned_histogram'][region]
                else: pre_bin_hists = pickle.load(f)['binned_histogram']
        else:
            with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/hist-pre{suffix}.pkl', 'rb') as f:
                if regional : pre_bin_hists = pickle.load(f)['binned_histogram'][region]
                else: pre_bin_hists = pickle.load(f)['binned_histogram']
        pre_box_stats = {}
        for k, counts in pre_bin_hists.items():
            stats = box_stats_from_hist(counts, residual_bins)
            if stats:
                pre_box_stats[k] = stats
        res['pre_' + f'{_min}'] = pre_box_stats
        colors_dict['pre_' + f'{_min}'] = pre_colors[i]


        # Load post-kriging binned histograms
        for j, model_name in enumerate(models):
            if model_name == 'best_for_each': continue
            with open(f"{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/nico_film/2020/17997535-1_17997535-2_17997535-3/{model_name}-hist-post{suffix}.pkl", 'rb') as f:
                if regional : post_bin_hists = pickle.load(f)['binned_histogram'][region]
                else: post_bin_hists = pickle.load(f)['binned_histogram']
            post_box_stats = {}
            for k, counts in post_bin_hists.items():
                stats = box_stats_from_hist(counts, residual_bins)
                if stats:
                    post_box_stats[k] = stats
            res[model_name + f'_{_min}'] = post_box_stats
            colors_dict[model_name + f'_{_min}'] = post_colors[i * len(models) + j]

    # Add a line for y = 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.75, alpha=0.5)

    # Plot all of them
    sorted_bins = sorted(res['pre_400'].keys(), key=numeric_key)
    labels = sorted_bins
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    n_models = len(res)
    width = 0.8 / n_models
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, n_models)

    for bin in sorted_bins :
        for i, (model_name, box_stats) in enumerate(res.items()):
            if bin in box_stats:
                draw_box(ax, x[labels.index(bin)] + offsets[i], box_stats[bin], color=colors_dict[model_name], width=width)


    ax.set_xticks(x)
    ax.set_xticklabels(labels) #, rotation=45)
    ax.set_ylabel("Residuals")
    ax.set_xlabel("AGBD bins [t/ha]")
    ax.legend(handles=[Rectangle((0,0),1,1, facecolor = colors_dict[model_name], edgecolor = 'black', alpha=0.7, label = model_name) for model_name in res.keys()])
    
    plt.tight_layout()
    plt.title(region)
    if regional : makedirs("figs/regional", exist_ok=True)
    plt.savefig(f"figs/{'regional/' if regional else ''}boxplot_residuals-{'-'.join(models)}{suffix}{f'-{region}' if regional else ''}.png", dpi=1200)


#######################################################################################################################
# Code execution

if __name__ == "__main__":

    # Parameters
    test = True # whether to use test set pre-kriging histograms
    subset = False # whether to use subsetted pre-kriging histograms
    regional = False # whether to plot regional histograms
    find_best = False # whether to find the best model for each region
    best_for_each = False # global boxplot using the best model for each region
    composites = True
    _min, _max = 500, 400

    # ref model for test: 45339599 for composites, 49714982 for non composites
    if composites : ref_model = '45339599' 
    else: ref_model = '49714982'

    # Global variables
    residual_bins = np.arange(-_min, _max, 1)
    suffix = f"{'_subset' if subset else ''}{'_test' if test else ''}{'_regional' if regional else ''}{('_' + str(_min) + '-' + str(_max)) if _min != 400 else ''}"

    # All of the models to compare to the pre-kriging baseline
    # to include the baselines : 'baseline_mean_std', 'baseline_all'
    models_to_compare = ['46171204', '46617715', '46637671', '1757019', '1617922', '1790043', '1820465', '47250199', '48428326'] # ['48428326'] # ['CCI', '48428326', '47250199', '47198254', '47118870', '46637671', '46617715', '46171204'] # ['47250199']
    colors = ['C0'] + all_colors[:len(models_to_compare)]
    colors_dict = {model: colors[i+1] for i, model in enumerate(models_to_compare)}

    # Best for each
    # regional_bests = {'California': '46783899', 'Cuba': '46783455', 'Paraguay': '46783455', 'UnitedRepublicofTanzania': '46783899', 'Ghana': '46783899', 'Austria': '46637671', 'Greece': '46783455', 'Nepal': '46783899', 'ShaanxiProvince': '46783455', 'NewZealand': '46637671', 'FrenchGuiana': '46765788'} # best over all bins
    regional_bests = {'California': '46783899', 'Cuba': '46783455', 'Paraguay': '46783455', 'UnitedRepublicofTanzania': '46783899', 'Ghana': '46783899', 'Austria': '46747815', 'Greece': '46783455', 'Nepal': '46783899', 'ShaanxiProvince': '46783899', 'NewZealand': '46783899', 'FrenchGuiana': '46171204'} # best over 200-250 and 250-300 bins
    if best_for_each :
        models_to_compare = list(set(regional_bests.values()))
        colors = ['C0'] + [f'C{i+1}' for i in range(len(models_to_compare))]
        colors_dict = {model: colors[i+1] for i, model in enumerate(models_to_compare)}
        colors_dict['best_for_each'] = 'pink'
        models_to_compare.append('best_for_each')

    if regional : regions = ['California', 'Cuba', 'Paraguay', 'UnitedRepublicofTanzania', 'Ghana', 'Austria', 'Greece', 'Nepal', 'ShaanxiProvince', 'NewZealand', 'FrenchGuiana']
    else: regions = ['global']

    if find_best and regional : regional_bests = {region: [] for region in regions}

    for region in regions :

        if regional : print(f"Processing region: {region}")

        elif best_for_each :
            post_stats = []
            global_count = {}
            # iterate over the regions, and for each get the best model
            for _region in ['California', 'Cuba', 'Paraguay', 'UnitedRepublicofTanzania', 'Ghana', 'Austria', 'Greece', 'Nepal', 'ShaanxiProvince', 'NewZealand', 'FrenchGuiana'] :
                _model = regional_bests[_region]
                with open(f"{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/nico_film/2020/17997535-1_17997535-2_17997535-3/{_model}-hist-post{suffix}_regional.pkl", 'rb') as f:
                    post_bin_hists = pickle.load(f)['binned_histogram'][_region]
                for k, counts in post_bin_hists.items():
                    if k not in global_count : global_count[k] = counts
                    else: global_count[k] += counts
            post_box_stats = {}
            for k, counts in global_count.items():
                stats = box_stats_from_hist(counts, residual_bins)
                if stats:
                    post_box_stats[k] = stats
            post_stats.append(( 'best_for_each', post_box_stats))
        
        # Load post-kriging binned histograms
        if not best_for_each : post_stats = []
        for model_name in models_to_compare :
            if model_name == 'CCI' :
                with open(f"{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/CCI/CCI-hist-post_test_500-400.pkl", 'rb') as f:
                    if regional : raise NotImplementedError("Regional CCI not implemented")
                    else: post_bin_hists = pickle.load(f)['binned_histogram']
            elif model_name == 'best_for_each' : continue
            else:
                with open(f"{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/nico_film/2020/17997535-1_17997535-2_17997535-3/{model_name}-hist-post{suffix}.pkl", 'rb') as f:
                    if regional : post_bin_hists = pickle.load(f)['binned_histogram'][region]
                    else: post_bin_hists = pickle.load(f)['binned_histogram']
            post_box_stats = {}
            for k, counts in post_bin_hists.items():
                stats = box_stats_from_hist(counts, residual_bins)
                if stats:
                    post_box_stats[k] = stats
            post_stats.append((model_name, post_box_stats))


        # Load pre-kriging binned histograms
        if test :
            with open(f"{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/hist-pre_test-{ref_model}{'_regional' if regional else ''}{('_' + str(_min) + '-' + str(_max)) if _min != 400 else ''}.pkl", "rb") as f:
                if regional : pre_bin_hists = pickle.load(f)['binned_histogram'][region]
                else: pre_bin_hists = pickle.load(f)['binned_histogram']
        else:
            with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/hist-pre{suffix}.pkl', 'rb') as f:
                if regional : pre_bin_hists = pickle.load(f)['binned_histogram'][region]
                else: pre_bin_hists = pickle.load(f)['binned_histogram']
        pre_box_stats = {}
        for k, counts in pre_bin_hists.items():
            stats = box_stats_from_hist(counts, residual_bins)
            if stats:
                pre_box_stats[k] = stats
        
        if find_best and regional :
            for bin in pre_box_stats.keys() :
                if bin not in ['200-250', '250-300'] : continue # only consider these bins for best model selection
                best = None
                best_model = None
                pre_median = pre_box_stats[bin]['median']
                for model_name, post_box_stats in post_stats:
                    # for each bin, compare the median residuals
                    if bin in post_box_stats:
                        post_median = post_box_stats[bin]['median']
                        if best is None or abs(post_median) < abs(best):
                            best = post_median
                            best_model = model_name
                regional_bests[region].append(best_model)

            continue

        #######################################################################################################################
        # Boxplot comparing the two    
        labels = []
        pre_data = []
        all_post_data = {}

        sorted_bins = sorted(pre_box_stats.keys(), key=numeric_key)
        if len(models_to_compare) > 0 :
            for model_name, post_box_stats in post_stats:
                post_data = []
                for bin in sorted_bins :
                    if bin in post_box_stats:
                        post_data.append(post_box_stats[bin])
                all_post_data[model_name] = post_data
        for bin in sorted_bins :
            if bin in pre_box_stats:
                labels.append(bin)
                pre_data.append(pre_box_stats[bin])

        # Plot settings
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(12, 6))
        n_post = len(all_post_data)
        n_total = 1 + n_post

        width = 0.05
        offsets = np.linspace(-0.35, 0.35, n_total)


        # Add a line for y = 0
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.75, alpha=0.5)

        # plot pre (shift left) and post (shift right)
        for i, prestats in enumerate(pre_data):
            draw_box(ax, x[i] + offsets[0], prestats, color=colors[0], width=width)  # Pre
        if len(models_to_compare) > 0 :
            for j, (model_name, poststats) in enumerate(all_post_data.items()):
                for i, stats in enumerate(poststats):
                    draw_box(ax, x[i] + offsets[j+1], stats, color = colors_dict[model_name], width=width)  # Post

        ax.set_xticks(x)
        ax.set_xticklabels(labels) #, rotation=45)
        ax.set_ylabel("Residuals")
        ax.set_xlabel("AGBD bins [t/ha]")
        ax.legend(handles=[Rectangle((0,0),1,1,facecolor=colors[0],edgecolor='black',alpha=0.6,label='Baseline')] +
                  [Rectangle((0,0),1,1,facecolor=colors_dict[model_name],edgecolor='black',alpha=0.6,label=name_to_id[model_name]) for model_name in all_post_data.keys()]
            #[Rectangle((0,0),1,1,facecolor=colors_dict[model_name],edgecolor='black',alpha=0.6,label=model_name + ' (best)' if (regional and model_name == regional_bests[region]) else model_name) for model_name in all_post_data.keys()]
        )
        plt.tight_layout()
        if region != 'global' : plt.title(region)
        if regional : makedirs("figs/regional", exist_ok=True)
        plt.savefig(f"figs/{'regional/' if regional else ''}boxplot_residuals-{'-'.join(models_to_compare)}{suffix}{f'-{region}' if regional else ''}.png", dpi=1200)
        print('Saved:', f"figs/{'regional/' if regional else ''}boxplot_residuals-{'-'.join(models_to_compare)}{suffix}{f'-{region}' if regional else ''}.png")

    if find_best and regional :
        # for each region, print the most frequent best model
        regional_bests = {region: max(set(models), key=models.count) for region, models in regional_bests.items()}
        print("Regional best models:", regional_bests)
    

    # For each model, get the median for each bin

    pre_kriging_medians = []
    for bin in sorted_bins :
        if bin in pre_box_stats :
            pre_kriging_medians.append(pre_box_stats[bin]['median'])


    for model_name, poststats in all_post_data.items():
        medians = []
        for i, stats in enumerate(poststats):
            medians.append(stats['median'])
        
        # Count how many times the post_kriging median is closer to 0 than the pre-kriging median
        count_better = sum(1 for pre_med, post_med in zip(pre_kriging_medians, medians) if abs(post_med) < abs(pre_med))
        print(model_name, count_better)

        # Sum the total improvement
        total_improvement = sum(abs(pre_med) - abs(post_med) for pre_med, post_med in zip(pre_kriging_medians, medians))
        print(model_name, total_improvement)
