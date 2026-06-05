"""

This script plots the boxplots of the binned RMSE for pre- and post-kriging results.
It creates the file `boxplot_rmses.png` with the comparison.

"""


#######################################################################################################################
# Imports

from config import DATA_ROOT
import pickle
import numpy as np
import matplotlib.pyplot as plt


#######################################################################################################################
# Code execution

if __name__ == "__main__":
    
    model_name = '42522619'

    #############################################################################################################################################################################
    # Pre-kriging

    print('Loading pre-kriging results...')

    with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/pre.pkl', 'rb') as f:
        pre_results = pickle.load(f)['overall']['binned']

    #############################################################################################################################################################################
    # Post-kriging

    print('Loading post-kriging results...')

    with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/nico_film/2020/17997535-1_17997535-2_17997535-3/{model_name}-post.pkl', 'rb') as f:
        post_results = pickle.load(f)['overall']['binned']
    
    #############################################################################################################################################################################

    # Get the RMSE for each bin
    bins = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500']
    pre_rmse = [pre_results[bin]['rmse'] for bin in bins]
    post_rmse = [post_results[bin]['rmse'] for bin in bins]

    num_footprints = [pre_results[bin]['num_footprints'] for bin in bins]

    # Boxplot for each bin, comparing pre and post
    x = np.arange(len(bins))
    width = 0.35   # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pre_rmse, width, label='Pre', color='C0', alpha=0.6, edgecolor='black')
    rects2 = ax.bar(x + width/2, post_rmse, width, label='Post', color='C1', alpha=0.6, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("RMSE [t/ha]")
    ax.set_xlabel("AGBD bins [t/ha]")
    ax.set_title("RMSE by AGBD bin")
    ax.legend()
    plt.savefig('boxplot-rmses.png', dpi=1200)