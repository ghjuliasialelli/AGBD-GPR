"""

This script generates per-region and per-biome density plots.


"""


#######################################################################################################################
# Imports

from config import DATA_ROOT
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

regions = ['California', 'Cuba', 'Paraguay', 'UnitedRepublicofTanzania', 'Ghana', 'Austria', 'Greece', 'Nepal', 'ShaanxiProvince', 'NewZealand', 'FrenchGuiana']
continents = ['Europe', 'Australasia', 'Africa', 'South Asia', 'South America', 'North America']
region_to_continent = {'California': 'North America', 'Cuba': 'North America', 'Paraguay': 'South America', 'UnitedRepublicofTanzania': 'Africa', 
                'Ghana': 'Africa', 'Austria': 'Europe', 'Greece': 'Europe', 'Nepal': 'South Asia', 'ShaanxiProvince': 'South Asia', 
                'NewZealand': 'Australasia', 'FrenchGuiana': 'South America'}
continent_to_numbers = {'North America': 1, 'South America': 2, 'Europe': 3, 'Africa': 4, 'South Asia': 5, 'North Asia': 6, 'Australasia': 7}
number_to_continent = {v: k for k, v in continent_to_numbers.items()}


#######################################################################################################################
# Code execution

if __name__ == "__main__":
    
    model_name = '48428326' # '47250199' or 'CCI or '48428326'
    test = True
    all = True

    if model_name != 'CCI': dir = f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/nico_film/2020/17997535-1_17997535-2_17997535-3'
    else: dir = f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/CCI'

    # Load the results
    if model_name == 'pre' : 
        fname = f"48428326-density{'_test' if test else ''}.pkl"
        _key = 'pre'
    else: 
        fname = f"{model_name}-density{'_test' if test else ''}.pkl"
        _key = 'post'
    with open(f'{dir}/{fname}', 'rb') as f: results = pickle.load(f)
    ref, post, biome, region = results['ref'], results[_key], results['biome'], results['region']
    ref, post, biome, region = np.array(ref), np.array(post), np.array(biome), np.array(region)
    
    ticks = np.arange(0, 501, 50)

    # Aggregated ###########################

    r2, rmse, mae, me = r2_score(ref, post), np.sqrt(mean_squared_error(ref, post)), mean_absolute_error(ref, post), np.mean(post - ref)
    slope = np.cov(ref, post)[0,1] / np.var(ref)

    plt.hist2d(ref.squeeze(), post.squeeze(), bins=(100, 100), cmap='Greens', norm=LogNorm(vmin=0.1, vmax=1000000))
    plt.colorbar(label='Number of samples')
    plt.plot([0, 500], [0, 500], 'k--')
    plt.xlim((0, 500))
    plt.ylim((0, 500))
    plt.xlabel('GEDI reference AGBD [Mg/ha]', fontsize=16)
    plt.ylabel('Predicted AGB [Mg/ha]', fontsize=16)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.gca().set_aspect('equal')

    # add text box with metrics
    textstr = '\n'.join((
        f'R2: {r2:.3f}',
        f'RMSE: {rmse:.2f} Mg/ha',
        f'MAE: {mae:.2f} Mg/ha',
        f'ME: {me:.2f} Mg/ha',
        f'Slope: {slope:.3f}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.savefig(f'figs/density/overall-{model_name}.png')
    plt.clf()


    if model_name != 'CCI' and all :

        # Per region ###########################
        for continent in continents :
            
            # Filter the data
            continent_number = continent_to_numbers[continent]
            mask = (region == continent_number)
            cont_ref, cont_post = ref[mask], post[mask]
            
            # Create the plot
            plt.hist2d(cont_ref.squeeze(), cont_post.squeeze(), bins=(100, 100), cmap='Greens', norm=LogNorm())
            plt.colorbar(label='Number of samples')
            plt.plot([0, 500], [0, 500], 'k--')
            plt.xlim((0, 500))
            plt.ylim((0, 500))
            plt.xlabel('GEDI reference AGBD [Mg/ha]', fontsize=16)
            plt.ylabel('Predicted AGB [Mg/ha]', fontsize=16)
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.tick_params(labelsize=12)
            plt.grid()
            plt.gca().set_aspect('equal')
            plt.title(continent, fontsize=18)
            plt.savefig(f'figs/density/region/{continent}-{model_name}.png')
            plt.clf()

        # Per biome ############################
        for b in REF_BIOMES.keys() :
            
            # Filter the data
            biome_int = int(b)
            mask = (biome == biome_int)
            bio_ref, bio_post = ref[mask], post[mask]
            
            # Create the plot
            plt.hist2d(bio_ref.squeeze(), bio_post.squeeze(), bins=(100, 100), cmap='Greens', norm=LogNorm())
            plt.colorbar(label='Number of samples')
            plt.plot([0, 500], [0, 500], 'k--')
            plt.xlim((0, 500))
            plt.ylim((0, 500))
            plt.xlabel('GEDI reference AGBD [Mg/ha]', fontsize=16)
            plt.ylabel('Predicted AGB [Mg/ha]', fontsize=16)
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.tick_params(labelsize=12)
            plt.grid()
            plt.gca().set_aspect('equal')
            plt.title(REF_BIOMES[b], fontsize=18)
            plt.savefig(f'figs/density/biome/{REF_BIOMES[b]}-{model_name}.png')
            plt.clf()