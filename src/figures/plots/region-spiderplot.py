"""

This script generates a spider plot comparing pre-kriging and post-kriging RMSE values across different continents.
It generates the region-spliderplot.png file.

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
    
    model_name = '47250199'
    test = True

    regions = ['California', 'Cuba', 'Paraguay', 'UnitedRepublicofTanzania', 'Ghana', 'Austria', 'Greece', 'Nepal', 'ShaanxiProvince', 'NewZealand', 'FrenchGuiana']
    continents = ['Europe', 'Australasia', 'Africa', 'South Asia', 'South America', 'North America']
    region_to_continent = {'California': 'North America', 'Cuba': 'South America', 'Paraguay': 'South America', 'UnitedRepublicofTanzania': 'Africa', 
                     'Ghana': 'Africa', 'Austria': 'Europe', 'Greece': 'Europe', 'Nepal': 'South Asia', 'ShaanxiProvince': 'South Asia', 
                     'NewZealand': 'Australasia', 'FrenchGuiana': 'South America'}

    #############################################################################################################################################################################
    # Pre-kriging

    print('Loading pre-kriging results...')

    suffix = ('_test' if test else '') + f'-{model_name}'
    with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/pre{suffix}.pkl', 'rb') as f:
        pre_kriging_results = pickle.load(f)

    regional_results = {continent: {'rmse': [], 'num_footprints': []} for continent in continents}
    for region in regions :

        region_results = pre_kriging_results[region]
        rmse = region_results['overall']['rmse']
        num_footprints = region_results['overall']['num_footprints']
        continent = region_to_continent[region]

        regional_results[continent]['rmse'].extend(rmse)
        regional_results[continent]['num_footprints'].extend(num_footprints)

    pre_results = {}
    for continent in continents :
        rmses = np.array(regional_results[continent]['rmse'])
        num_footprints = np.array(regional_results[continent]['num_footprints'])
        total_footprints = np.sum(num_footprints)
        if total_footprints > 0 :
            num_footprints = num_footprints / total_footprints
            total_rmse = np.sqrt(np.sum(num_footprints * np.power(rmses, 2)))
        else :
            print('No footprints for continent:', continent)
            total_rmse = np.nan
        pre_results[continent] = total_rmse


    #############################################################################################################################################################################
    # Post-kriging

    print('Loading post-kriging results...')

    suffix = '_test' if test else ''
    with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/nico_film/2020/17997535-1_17997535-2_17997535-3/{model_name}-post{suffix}.pkl', 'rb') as f:
        post_kriging_results = pickle.load(f)
    
    regional_results = {continent: {'rmse': [], 'num_footprints': []} for continent in continents}
    for region in regions :

        region_results = post_kriging_results[region]
        rmse = region_results['overall']['rmse']
        num_footprints = region_results['overall']['num_footprints']
        continent = region_to_continent[region]

        regional_results[continent]['rmse'].extend(rmse)
        regional_results[continent]['num_footprints'].extend(num_footprints)

    post_results = {}
    for continent in continents :
        rmses = np.array(regional_results[continent]['rmse'])
        num_footprints = np.array(regional_results[continent]['num_footprints'])
        total_footprints = np.sum(num_footprints)
        if total_footprints > 0 :
            num_footprints = num_footprints / total_footprints
            total_rmse = np.sqrt(np.sum(num_footprints * np.power(rmses, 2)))
        else :
            print('No footprints for continent:', continent)
            total_rmse = np.nan
        post_results[continent] = total_rmse


    #############################################################################################################################################################################
    # Plotting

    pre_values = [pre_results[continent] for continent in continents]
    print(pre_values)
    post_values = [post_results[continent] for continent in continents]
    print(post_values)

    categories = continents
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
    #ax.set_rlabel_position(75)
    #plt.yticks([20, 40, 60, 80, 100, 120], ["20", "40", "60", "80", "100", "120"], color="gray", size=14)
    #plt.ylim(20, 130)

    # To align with the biome spiderplot
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
    plt.savefig('region-spiderplot.png', dpi=1200)