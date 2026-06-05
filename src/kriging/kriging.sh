#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --output=${DATA_ROOT}/logs/kriging-%A_%a.out
#SBATCH --error=${DATA_ROOT}/logs/kriging-%A_%a.out
#SBATCH --mem-per-cpu=9G
#SBATCH --array=1,107,120,,565
#SBATCH --gpus=1
#SBATCH --gres=gpumem:15g


##################################################################################################################
# Main settings

seed=10

# Overall parameters
year=2020
arch='nico_film'
ens_models=('17997535-1' '17997535-2' '17997535-3')
composites="true"

# Training parameters
test_holdout=0.3
val_holdout=0.2
max_train_footprints=25000 # 10000 25000
mem_max_footprints=25000
ref_model="48428326"
max_split_diff=15
max_tries=100
stripe_size=50
ood="true"

# Model parameters
aux="STD" # auxiliary variable: STD or none
extra_features=("mean_25" "std_25" "cv_25" "lr_25") #("mean_9" "std_9" "cv_9" "lr_9" "sobel" "laplace") # whether to compute additional features (none, median, mean)
pred_vals="true" # whether to use the predicted values as additional auxiliary data
norm_aux="min_max" # normalize auxiliary data: min_max or false
norm_res="true" # whether to normalize the output (will apply z-scoring)
coords="true" # whether to use the x/y coordinates
norm_coords="true" # whether to normalize the x/y coordinates (will apply min-max scaling)
lr=1e-2 # learning rate
num_iterations=1000
pos_loss="true" # whether to ensure positive loss
agb="false" # whether to use kriging on the AGB values rather than the residuals

x_lengthscale="dynamic"
y_lengthscale="dynamic"
fix_x_y="true" # whether to fix the x/y lengthscales to the provided values
z_aux_lengthscale=50.0
z_pred_lengthscale=50.0
outputscale=100.0
gaussian_noise=10.0
learned_noise=10.0

matern_nu=0.5 # 0.5, 1.5, 2.5


COMPUTE_VAR="true"
SAVE="true" # save the metrics .pkl files
SAVE_preds="false" # save the prediction .tif files

offline="true" # run wandb in offline mode (no network sync) to avoid hangs without internet

# Set the correct values if the auxiliary data is processed
if [[ "$norm_aux" != "false" ]]; then  # min-max scaling
    learned_noise=1.0
    z_aux_lengthscale=0.1
    z_pred_lengthscale=0.1
    eft_lengthscale=0.1
fi

# Scale the x/y lengthscales
# condition if norm_coords is True and x_lengthscale and y_lengthscale are not dynamic
if [[ "$norm_coords" == "true" && "$x_lengthscale" != "dynamic" && "$y_lengthscale" != "dynamic" ]]; then
    x_lengthscale=$(echo "$x_lengthscale / 10980" | bc -l)
    y_lengthscale=$(echo "$y_lengthscale / 10980" | bc -l)
fi

# Set the correct values if the residuals are standardized
if [[ "$norm_res" == "true" ]]; then
    outputscale=1.0
    gaussian_noise=0.1
    patience=50
    min_delta=0.001
else
    patience=5
    min_delta=0.001
fi




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
    source ${DATA_ROOT}/mambaforge/etc/profile.d/conda.sh
    conda activate krige
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

##### When we want to run tests locally
# 10TEL 497
# 30PWS 250
# 21JWM 1
# 45RXL 410
if [[ "$first_part" == "scratch3" ]]; then
    composites="false"
    s2_tile="21JWM"
    model_name="local-1"
    SAVE_preds="false"
fi
#####



echo "Launching kriging for tile: " ${s2_tile}


# If "subset" in LIST_TILES_FILE, set subset=True
# i.e. if we use the pre-defined subset of tiles, we still want them to be indexed as in the full list of tiles
if [[ "$LIST_TILES_FILE" == *"subset"* ]]; then
    ref_file="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/inference/per_tile/valid_${year}.txt"
    # Find the index of s2_tile in ref_file
    line_num=$(grep -nx "$s2_tile" "$ref_file" | cut -d: -f1)
    model_name=${model_name%%-*}-${line_num}
fi

# For a few tiles, use fixed lengthscales
if [[ "$x_lengthscale" == "dynamic" || "$y_lengthscale" == "dynamic" ]]; then
    if [[ "$s2_tile" == "48SXD" || "$s2_tile" == "49SBT" || "$s2_tile" == "49SCT" ]]; then
        x_lengthscale=5000
        y_lengthscale=5000
        fix_x_y="true"
        echo "Using fixed lengthscales of 5000 for tile ${s2_tile}"
        if [[ "$norm_coords" == "true" ]]; then
            x_lengthscale=$(echo "$x_lengthscale / 10980" | bc -l)
            y_lengthscale=$(echo "$y_lengthscale / 10980" | bc -l)
        fi
    fi
fi

##################################################################################################################
# Launch the correction

python kriging.py \
    --s2_tile $s2_tile --year $year --arch $arch --ens_models ${ens_models[@]} \
    --test_holdout $test_holdout --val_holdout $val_holdout --stripe_size $stripe_size --path_predictions $path_predictions \
    --path_gedi $path_gedi --path_geometries $path_geometries --path_dem $path_dem --path_kriging $path_kriging \
    --aux $aux --extra_features ${extra_features[@]} --norm_aux $norm_aux --norm_coords $norm_coords \
    --norm_res $norm_res --coords $coords --pred_vals $pred_vals --matern_nu $matern_nu \
    --num_iterations $num_iterations --pos_loss $pos_loss --lr $lr --max_train_footprints $max_train_footprints \
    --mem_max_footprints $mem_max_footprints --ref_model $ref_model --x_lengthscale $x_lengthscale --y_lengthscale $y_lengthscale \
    --fix_x_y $fix_x_y --z_aux_lengthscale $z_aux_lengthscale --z_pred_lengthscale $z_pred_lengthscale --eft_lengthscale $eft_lengthscale \
    --outputscale $outputscale --gaussian_noise $gaussian_noise --learned_noise $learned_noise --model_name $model_name \
    --patience $patience --min_delta $min_delta --COMPUTE_VAR $COMPUTE_VAR --SAVE $SAVE \
    --SAVE_preds $SAVE_preds --max_split_diff $max_split_diff --max_tries $max_tries --seed $seed \
    --composites $composites --ood $ood --agb $agb --offline $offline
                        --offline $offline