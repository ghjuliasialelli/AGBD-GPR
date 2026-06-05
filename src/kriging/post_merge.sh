#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --output=${DATA_ROOT}/logs/esa-%A_%a.out
#SBATCH --error=${DATA_ROOT}/logs/esa-%A_%a.out
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:15g
#SBATCH --array=1-111

##################################################################################################################
# Main settings

inf_model="17997535-1_17997535-2_17997535-3"
krig_model="47250199"
arch="nico_film"
buffer_size=250
year=2020
crs=4326 # 8857 (for Equal Earth) 4326 (for WGS84, the ESA one)
border_crop=20
method="cosine" # gaussian, cosine
compute_metrics="false"
force="true"
dtype="uint16" # uint16 or float32

# Tiles to run the inference on
LIST_TILES_FILE="EcosystemAnalysis/Models/Biomes/kriging/txt_files/ESA_${year}.txt"
echo "Will read products from file ${LIST_TILES_FILE}"

# If enabled, launch merge only for the tiles in LIST_FAILS
SKIP_TILES="false"
LIST_FAILS="S45E165 S45E168 S48E165 S48E168"

##################################################################################################################
# Establish the paths based on whether we're on the cluster or not

current_directory=$(pwd)
echo "Current Directory: $current_directory"

first_part=$(echo "$current_directory" | cut -d'/' -f2)

if [[ "$first_part" == "cluster" ]]; then
    echo "Running on a cluster"
    source activate krige
    BASE_PATH_DATA="${DATA_ROOT}/Data"
    BASE_PATH_CODE="${DATA_ROOT}"
    path_mask="${DATA_ROOT}/data/ESA_WorldCover/data/3deg_cogs"
    temp_dir="${TMPDIR}"
elif [[ "$first_part" == "scratch3" ]]; then
    echo "Running on a local machine"
    SLURM_ARRAY_TASK_MIN=1
    SLURM_ARRAY_TASK_MAX=1
    # SLURM_ARRAY_TASK_ID is not set in this case, so we need to set it manually
    if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
        echo "SLURM_ARRAY_TASK_ID is not set; defaulting to 1"
        SLURM_ARRAY_TASK_ID=1
    else
        echo "SLURM_ARRAY_TASK_ID is set to $SLURM_ARRAY_TASK_ID"
    fi
    BASE_PATH_DATA="${DATA_ROOT}"
    BASE_PATH_CODE="${DATA_ROOT}"
    path_mask="${DATA_ROOT}/WorldCover"
    temp_dir='tmp'
else
    echo "Environment unknown"
fi

# Define the paths to the data
path_esa="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/helper/3x3/esa_worldcover_tiles.geojson"
path_predictions="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/kriging/predictions/"
path_gedi="${BASE_PATH_DATA}/GEDI/L4A_California_Cuba_Paraguay_UnitedRepublicofTanzania_Ghana_Austria_Greece_Nepal_ShaanxiProvince_NewZealand_FrenchGuiana.shp"
path_geometries="${BASE_PATH_CODE}/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"
path_AOIs="${BASE_PATH_CODE}/BiomassDatasetCreation/Data/countrySelection/AOIs.geojson"
LIST_PRODS_FILE="${BASE_PATH_CODE}/${LIST_TILES_FILE}"

################################################################################################################################
# Parse the tile names from LIST_PRODS_FILE

readarray -t tile_names < ${LIST_PRODS_FILE}
num_tiles=${#tile_names[@]}

if [[ "$first_part" == "cluster" ]]; then
    # Check if SLURM_ARRAY_TASK_MIN is 1
    if [ "$SLURM_ARRAY_TASK_MIN" -ne 1 ]; then
        echo "Assertion failed: SLURM_ARRAY_TASK_MIN is not equal to 1" >&2
        exit 1
    fi
    # Check if SLURM_ARRAY_TASK_MAX is equal to the length of the array
    if [ "$SLURM_ARRAY_TASK_MAX" -ne "$num_tiles" ]; then
        echo "Assertion failed: SLURM_ARRAY_TASK_MAX is not equal to the length of the array" >&2
        exit 1
    fi
fi

# Select the i-th element in the array, where i is the current job number - 1 (SLURM_ARRAY_TASK_ID is 1-indexed)
esa_tile=${tile_names[$SLURM_ARRAY_TASK_ID-1]}

# Check if the tile should be skipped
to_skip="S45W177 S51E165 S54E168" # tiny islands
if [[ " $to_skip " =~ " $esa_tile " ]]; then
    echo "Tile ${esa_tile} is in the skip list. Skipping this tile." 
    exit 0
fi

# Local debugging
if [[ "$first_part" == "scratch3" ]]; then
    esa_tile='N06W006'
fi

# If SKIP_TILES is enabled, continue only if the tiles is in LIST_FAILS
if [[ "$SKIP_TILES" == "true" ]]; then
    if [[ ! " ${LIST_FAILS[@]} " =~ " ${esa_tile} " ]]; then
        echo "Tile ${esa_tile} is not in the list of failed tiles. Skipping this tile."
        exit 0
    fi
fi

echo "Launching merge for ESA tile: " ${esa_tile}
echo "with method: " ${method}

##################################################################################################################
# Launch the correction

python post_merge.py --ESA ${path_esa} --GEDI ${path_gedi} --S2 ${path_geometries} \
    --predictions ${path_predictions} --inf_model ${inf_model} --krig_model ${krig_model} \
    --tile_name ${esa_tile} --arch ${arch} --buffer_size ${buffer_size} --year ${year} \
    --crs ${crs} --border_crop ${border_crop} --method ${method} --compute_metrics ${compute_metrics} \
    --AOIs ${path_AOIs} --mask ${path_mask} --tmp_dir ${temp_dir} --force ${force} --dtype ${dtype}