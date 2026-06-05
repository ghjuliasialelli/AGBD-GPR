#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=${DATA_ROOT}/logs/threshold-%A_%a.out
#SBATCH --error=${DATA_ROOT}/logs/threshold-%A_%a.out
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1,6,12,13,15,59,64,95,105,136,142,155,158,174,202,211,230,260,270,272,305,326,329,359,410,450,497,531,565

# If enabled, relaunch the kriging jobs for the provided relaunch_id, only for the tiles in LIST_FAILS
RELAUNCH_FAILS="true"
relaunch_id="45747788"

# If enabled, launch kriging only for the tiles in LIST_FAILS
SKIP_TILES="false"

# Tiles for which to run the kriging if either RELAUNCH_FAILS or SKIP_TILES is true
LIST_FAILS="45RTM 44RMU 10SGH 11SLA 33TUM 11SKB 45RVM 44RPS 45RWL 44RQS 45RUM 45RVL 10TEM 32TQT 44RPT 33TUN 44RQT 45RTN 44RNU 44RNT 20KQB 44RPU 32TPT 45RXL 11SKC 10TEL 10SGG"

##################################################################################################################
# Main settings

seed=10

# Overall parameters
year=2020
arch='nico_film'
ens_models=('17997535-1' '17997535-2' '17997535-3')
composites="true"
ood="false"
SAVE="true" # save the metrics .pkl files
SAVE_preds="true" # save the prediction .tif files

# Tiles to run the inference on
LIST_TILES_FILE="EcosystemAnalysis/Models/Biomes/inference/per_tile/valid_${year}.txt"
echo "Will read products from file ${LIST_TILES_FILE}"

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
    model_name=${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
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
    model_name="local-${SLURM_ARRAY_TASK_ID}"
else
    echo "Environment unknown"
fi

# Define the paths to the data
path_predictions="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/predictions/"
path_gedi="${BASE_PATH_DATA}/GEDI/L4A_California_Cuba_Paraguay_UnitedRepublicofTanzania_Ghana_Austria_Greece_Nepal_ShaanxiProvince_NewZealand_FrenchGuiana-indexed.gpkg"
path_geometries="${BASE_PATH_CODE}/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"
path_kriging="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/kriging"
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
s2_tile=${tile_names[$SLURM_ARRAY_TASK_ID-1]}

# Check if the tile should be skipped
to_skip="58FEJ 58FEK 59FLB 01GEM 60FXL 58FGG 60FXK 10SDG 59GNQ 58GFN 11SKS 49SET 11SMR 31NCG 35MQN 45RWM 22NCM 30PVS 49SEC 17QQC 17QQF 11SQV 59HQU 37MCT"
if [[ " $to_skip " =~ " $s2_tile " ]]; then
    echo "Tile ${s2_tile} is in the skip list. Skipping this tile." 
    exit 0
fi

# If RELAUNCH_FAILS is enabled, continue only if the tile is in LIST_FAILS
if [[ "$RELAUNCH_FAILS" == "true" ]]; then
    if [[ ! " ${LIST_FAILS[@]} " =~ " ${s2_tile} " ]]; then
        echo "Tile ${s2_tile} is not in the list of failed tiles. Skipping this tile."
        exit 0
    fi
fi

# if SKIP_TILES is enabled, continue only if the tiles is in LIST_FAILS
if [[ "$SKIP_TILES" == "true" ]]; then
    if [[ ! " ${LIST_FAILS[@]} " =~ " ${s2_tile} " ]]; then
        echo "Tile ${s2_tile} is not in the list of failed tiles. Skipping this tile."
        exit 0
    fi
fi


echo "Launching predictions for tile: " ${s2_tile}


# If "subset" in LIST_TILES_FILE, set subset=True
# i.e. if we use the pre-defined subset of tiles, we still want them to be indexed as in the full list of tiles
if [[ "$LIST_TILES_FILE" == *"subset"* ]]; then
    ref_file="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/inference/per_tile/valid_${year}.txt"
    # Find the index of s2_tile in ref_file
    line_num=$(grep -nx "$s2_tile" "$ref_file" | cut -d: -f1)
    model_name=${model_name%%-*}-${line_num}
fi

##################################################################################################################
# Launch the correction

python threshold.py --s2_tile $s2_tile --year $year --arch $arch --ens_models ${ens_models[@]} \
                    --path_predictions $path_predictions --path_gedi $path_gedi --path_geometries $path_geometries --path_kriging $path_kriging \
                    --SAVE $SAVE --seed $seed --composites $composites --SAVE_preds $SAVE_preds --ood $ood