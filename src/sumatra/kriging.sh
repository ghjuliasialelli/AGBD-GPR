#!/bin/bash

##################################################################################################################
# Main settings

reference="gedi" # gedi (Mozambique field-reference path removed)
pred="ours" # ours or CCI
downsampled="true" # when pred=ours: use our downsampled merged prediction (.tif) instead of the full-res S2 directory
stripe_size=50

s2_tile="N/A"
echo "Launching predictions for ${pred} Sumatra map, kriging on ${reference} data."

x_lengthscale="dynamic"
y_lengthscale="dynamic"

##################################################################################################################
# General kriging settings

seed=10

# Overall parameters
year=2020
arch='nico_film'
ens_models=('17997535-1' '17997535-2' '17997535-3')
composites="true"

# The downsampled prediction we krige on is the composite one; non-composite is not supported here
if [[ "$pred" == "ours" && "$downsampled" == "true" && "$composites" != "true" ]]; then
    echo "Error: downsampled=true with pred=ours requires composites=true (we krige on merged_downsampled-100m_composite.tif)."
    exit 1
fi

# Training parameters
test_holdout=0.3
val_holdout=0.2
max_train_footprints=25000 # 10000 25000
max_split_diff=15
max_tries=100
ood="false"

# Model parameters
aux="STD" # auxiliary variable: STD or none
extra_features=("mean_25" "std_25" "cv_25" "lr_25" "sobel" "laplace") #("mean_9" "std_9" "cv_9" "lr_9" "sobel" "laplace") # whether to compute additional features
pred_vals="true" # whether to use the predicted values as additional auxiliary data
norm_aux="min_max" # normalize auxiliary data: min_max or false
norm_res="true" # whether to normalize the output (will apply z-scoring)
coords="true" # whether to use the x/y coordinates
norm_coords="true" # whether to normalize the x/y coordinates (will apply min-max scaling)
lr=1e-2 # learning rate
num_iterations=1000
pos_loss="true" # whether to ensure positive loss
agb="false" # whether to use kriging on the AGB values rather than the residuals

fix_x_y="true" # whether to fix the x/y lengthscales to the provided values
z_aux_lengthscale=50.0
z_pred_lengthscale=50.0
outputscale=100.0
gaussian_noise=10.0
learned_noise=10.0

matern_nu=0.5 # 0.5, 1.5, 2.5


COMPUTE_VAR="true"
SAVE="true" # save the metrics .pkl files
SAVE_preds="true" # save the prediction .tif files

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



##################################################################################################################
# Establish the paths

BASE_PATH_DATA="${DATA_ROOT}"
BASE_PATH_CODE="${DATA_ROOT}"

path_geometries="${BASE_PATH_CODE}/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"
path_dem="${BASE_PATH_DATA}/ALOS"
path_kriging="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/Sumatra"

if [[ "$pred" == "CCI" ]]; then
    path_predictions="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/Sumatra/Data/CCI/CCI_N00E100.tif"
    model_name="sumatra_cci_${reference}"
elif [[ "$pred" == "ours" ]]; then
    if [[ "$downsampled" == "true" ]]; then
        ens_join=$(IFS=_; echo "${ens_models[*]}")
        path_predictions="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/Sumatra/Data/merged/${ens_join}/merged_downsampled-100m_composite.tif"
        model_name="sumatra_downsampled_${reference}"
    else
        path_predictions="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/Sumatra/Data/S2"
        model_name="sumatra_${reference}"
    fi
elif [[ "$pred" == "gdbt" ]]; then
    path_predictions="${BASE_PATH_CODE}/EcosystemAnalysis/Models/Biomes/Sumatra/Data/gdbt.tif"
    model_name="sumatra_gdbt_${reference}"
else
    echo "Error: pred must be 'CCI', 'ours' or 'gdbt'."
    exit 1
fi

# if composites, add _composite to the model name
if [[ "$composites" == "true" ]]; then
    model_name="${model_name}_composite"
fi

##################################################################################################################
# Launch the correction

if [[ "$reference" == "gedi" ]]; then

    path_gedi="${BASE_PATH_DATA}/GEDI/Sumatra/L4A_Sumatra.gpkg"

    if [[ "$pred" == "ours" ]]; then

        # Use the downsampled merged prediction (.tif) if requested, else the full-res S2 directory
        if [[ "$downsampled" == "true" ]]; then ours_script="kriging_downsampled_gedi.py"; else ours_script="kriging_ours_gedi.py"; fi

        python $ours_script \
        --s2_tile $s2_tile --year $year --arch $arch --ens_models ${ens_models[@]} \
        --test_holdout $test_holdout --val_holdout $val_holdout --stripe_size $stripe_size --path_predictions $path_predictions \
        --path_gedi $path_gedi --path_geometries $path_geometries --path_dem $path_dem --path_kriging $path_kriging \
        --aux $aux --extra_features ${extra_features[@]} --norm_aux $norm_aux --norm_coords $norm_coords \
        --norm_res $norm_res --coords $coords --pred_vals $pred_vals --matern_nu $matern_nu \
        --num_iterations $num_iterations --pos_loss $pos_loss --lr $lr --max_train_footprints $max_train_footprints \
        --x_lengthscale $x_lengthscale --y_lengthscale $y_lengthscale --fix_x_y $fix_x_y --z_aux_lengthscale $z_aux_lengthscale \
        --z_pred_lengthscale $z_pred_lengthscale --eft_lengthscale $eft_lengthscale --outputscale $outputscale --gaussian_noise $gaussian_noise \
        --learned_noise $learned_noise --model_name $model_name --patience $patience --min_delta $min_delta \
        --COMPUTE_VAR $COMPUTE_VAR --SAVE $SAVE --SAVE_preds $SAVE_preds --max_split_diff $max_split_diff \
        --max_tries $max_tries --seed $seed --composites $composites --ood $ood \
        --agb $agb

    else

        python kriging_gedi.py \
        --s2_tile $s2_tile --year $year --arch $arch --ens_models ${ens_models[@]} \
        --test_holdout $test_holdout --val_holdout $val_holdout --stripe_size $stripe_size --path_predictions $path_predictions \
        --path_gedi $path_gedi --path_geometries $path_geometries --path_dem $path_dem --path_kriging $path_kriging \
        --aux $aux --extra_features ${extra_features[@]} --norm_aux $norm_aux --norm_coords $norm_coords \
        --norm_res $norm_res --coords $coords --pred_vals $pred_vals --matern_nu $matern_nu \
        --num_iterations $num_iterations --pos_loss $pos_loss --lr $lr --max_train_footprints $max_train_footprints \
        --x_lengthscale $x_lengthscale --y_lengthscale $y_lengthscale --fix_x_y $fix_x_y --z_aux_lengthscale $z_aux_lengthscale \
        --z_pred_lengthscale $z_pred_lengthscale --eft_lengthscale $eft_lengthscale --outputscale $outputscale --gaussian_noise $gaussian_noise \
        --learned_noise $learned_noise --model_name $model_name --patience $patience --min_delta $min_delta \
        --COMPUTE_VAR $COMPUTE_VAR --SAVE $SAVE --SAVE_preds $SAVE_preds --max_split_diff $max_split_diff \
        --max_tries $max_tries --seed $seed --composites $composites --ood $ood \
        --agb $agb
    
    fi


else
    echo "Error: reference must be 'field' or 'gedi'."
    exit 1
fi

