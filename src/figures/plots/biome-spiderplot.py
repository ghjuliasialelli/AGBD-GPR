"""

This script generates a spider plot comparing pre-kriging and post-kriging RMSE values across different biomes.
It generates the biome-spliderplot.png file.

"""


#######################################################################################################################
# Imports

from config import DATA_ROOT
import pickle
import numpy as np
import matplotlib.pyplot as plt

REF_BIOMES = {
    '20': 'Shrubs', 
    '30': 'HV', 
    '40': 'Crops', 
    '90': 'HW', 
    '111': 'C-ENL', 
    '112': 'C-EBL', 
    '114': 'C-DBL', 
    '115': 'C-M', 
    '116': 'C-O', 
    '121': 'O-ENL', 
    '122': 'O-EBL', 
    '124': 'O-DBL', 
    '125': 'O-M', 
    '126': 'O-O'
}

biomes = list([int(b) for b in REF_BIOMES.keys()])
biomes_labels = list(REF_BIOMES.values())

#######################################################################################################################
# Code execution

if __name__ == "__main__":
    
    model_name = '47250199'
    test = True    

    #############################################################################################################################################################################
    # Pre-kriging

    print('Loading pre-kriging results...')
    # pre_test-47250199_500-400.pkl
    suffix = ('_test' if test else '') + f'-{model_name}'
    with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/pre{suffix}.pkl', 'rb') as f:
        pre_results = pickle.load(f)['overall']['biome']

    #############################################################################################################################################################################
    # Post-kriging

    print('Loading post-kriging results...')
    # 47250199-post_test_500-400.pkl
    suffix = '_test' if test else ''
    with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/nico_film/2020/17997535-1_17997535-2_17997535-3/{model_name}-post{suffix}.pkl', 'rb') as f:
        post_results = pickle.load(f)['overall']['biome']
    

    #############################################################################################################################################################################
    # Plotting

    pre_values = [pre_results[biome]['rmse'] for biome in biomes]
    print(pre_values)
    post_values = [post_results[biome]['rmse'] for biome in biomes]
    print(post_values)

    categories = biomes_labels
    N = len(categories)

    # ======= Radar chart setup =======
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # complete loop

    # ======= Prepare data =======
    pre_values += pre_values[:1]
    post_values += post_values[:1]

    # ======= Plotting =======
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

    # Plot FILM
    ax.plot(angles, pre_values, color="#DE5BD894", linewidth=3, label='Pre-kriging')

    # Plot ENS
    ax.plot(angles, post_values, color="#0AA5E3CE", linewidth=3, label='Post-kriging')

    # ======= Aesthetics =======
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14)
    ax.xaxis.set_tick_params(pad=15)

    # Y-axis
    ax.set_rlabel_position(75)
    plt.yticks([0, 40, 80, 120, 160], ["0", "40", "80", "120", "160"], color="gray", size=14)
    plt.ylim(0, 180)

    # Legend
    #plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), borderaxespad=0.)

    legend = plt.legend(loc='upper left',
                        bbox_to_anchor=(0.3, 0.95),  # Adjust position here
                        frameon=True,
                        framealpha=0.9,
                        facecolor='white',
                        edgecolor='gray',
                        fontsize = 15)

    plt.tight_layout()
    plt.savefig('biome-spiderplot.png', dpi=1200)