#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --output=${DATA_ROOT}/logs/predict-%A_%a.out
#SBATCH --error=${DATA_ROOT}/logs/predict-%A_%a.out
#SBATCH --mem-per-cpu=9G
#SBATCH --array=1-565
#SBATCH --gpus=1
#SBATCH --gres=gpumem:15g

# If enabled, relaunch the kriging jobs for the provided relaunch_id, only for the tiles in LIST_FAILS
RELAUNCH_FAILS="false"
relaunch_id="46747815"

# If enabled, launch kriging only for the tiles in LIST_FAILS
SKIP_TILES="false"
if [ "$SKIP_TILES" == "true" ]; then
    SLURM_ARRAY_JOB_ID=47198254
fi

# Tiles for which to run the kriging if either RELAUNCH_FAILS or SKIP_TILES is true
LIST_FAILS="11SPU 20KQV 10SGJ 36MZS 11SMB 35SKA 59GQP 22NBH 10SFG 30NYM 49SBA 11SLC 22NCL 37LCL 21KTR 21KTU 21KXQ 59HPA 30PXR 36MWV 60HXD 11SQS 36MWD 35MRM 36LZQ 44RNT 33UVQ 33UVP 10SFH 21JXM 21JWN 21KUT 20KNC 49SCV 30NXL 36LZN 48SXC 35MRN 32TPT 36MYC 36MZC 36LYN 44RQR 18QWH 37MDR 49SCC 21JVL 21KWR 10TCK 10TDM 59GML 37MFM 34TGK 17QMF 30PYS 10SGF 44RNR 60HXC 44RPR 16QGK 36LUR 17QKD 20KPB 49SBT 18QTJ 17QQD 59HQB 37MDP 44RMS 36MUC 36MWB 34SGF"

##################################################################################################################
# Main settings

seed=10

# Parameters
kriging_model="phantasmal-threshold-11994"
year=2020
arch='nico_film'

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
    tile_id=${SLURM_ARRAY_TASK_ID}
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
path_dem="${BASE_PATH_DATA}/ALOS"
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
    # Edit the model name to include the relaunch ID
    model_name="${relaunch_id}-${SLURM_ARRAY_TASK_ID}"
fi

# if SKIP_TILES is enabled, continue only if the tiles is in LIST_FAILS
if [[ "$SKIP_TILES" == "true" ]]; then
    if [[ ! " ${LIST_FAILS[@]} " =~ " ${s2_tile} " ]]; then
        echo "Tile ${s2_tile} is not in the list of failed tiles. Skipping this tile."
        exit 0
    fi
fi

##### When we want to run tests locally
# 10TEL 497
# 30PWS 250
if [[ "$first_part" == "scratch3" ]]; then
    s2_tile="30PWS"
    tile_id=250
fi
#####

echo "Launching predictions for tile: " ${s2_tile}


# If "subset" in LIST_TILES_FILE, set subset=True
# i.e. if we use the pre-defined subset of tiles, we still want them to be indexed as in the full list of tiles
if [[ "$LIST_TILES_FILE" == *"subset"* ]]; then
    ref_file="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/inference/per_tile/valid_${year}.txt"
    # Find the index of s2_tile in ref_file
    line_num=$(grep -nx "$s2_tile" "$ref_file" | cut -d: -f1)
    tile_id=${line_num}
fi

##################################################################################################################
# Launch the correction

export WANDB_SERVICE_PORT=0

python predict.py --kriging_model ${kriging_model} \
    --tile_id ${tile_id} \
    --s2_tile ${s2_tile} \
    --year ${year} \
    --arch ${arch} \
    --path_predictions ${path_predictions} \
    --path_dem ${path_dem} \
    --path_kriging ${path_kriging} \
    --seed ${seed}