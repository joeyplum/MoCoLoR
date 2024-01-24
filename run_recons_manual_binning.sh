#!/bin/bash

# Replace 'python' with the appropriate command if 'python' doesn't work in your environment.
PYTHON_EXECUTABLE=python

# Function to run the binning_quantile.py script with given arguments
run_binning_manual() {
    $PYTHON_EXECUTABLE binning/binning_manual.py --fname /storage/Joey/MoCoLoR/data/floret-740H-032c/ --plot 1
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_nufft() {
    $PYTHON_EXECUTABLE recon_dcf_nufft_npy.py /storage/Joey/MoCoLoR/data/floret-740H-032c/ --lambda_TV 0 --vent_flag 0 --recon_res 80 --scan_res  220 --res_scale 0.7
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_xdgrasp1() {
    $PYTHON_EXECUTABLE recon_xdgrasp_npy.py /storage/Joey/MoCoLoR/data/floret-740H-032c/ --lambda_TV 0.05 --vent_flag 0 --recon_res 117 --scan_res  220
}

# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_mocolor1() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py /storage/Joey/MoCoLoR/data/floret-740H-032c/ --use_dcf 2 --lambda_lr 0.01 --vent_flag 1 --res_scale 0.5 --recon_res 50 --scan_res  220 --mr_cflag 1 --iner_iter 5 
}

# Generate a mask
run_segmentation() {
    $PYTHON_EXECUTABLE segmentation/segmentation_ute.py --fname /storage/Joey/MoCoLoR/data/floret-740H-032c/results/ --filename img_mocolor_10_bin_117_resolution.nii --plot 0 --mask 1
}

echo "Running binning_manual.py ..."
# run_binning_manual
echo "Finished binning_manual.py"

echo "Running recon_dcf_nufft_npy.py ..."
# run_recon_nufft
echo "Finished recon_dcf_nufft_npy.py"

echo "Running recon_xdgrasp_npy.py ..."
# run_recon_xdgrasp1
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_mocolor_npy.py ..."
run_recon_mocolor1
echo "Finished recon_mocolor_npy.py"

echo "Running segmentation_ute.py..."
# run_segmentation
echo "Finished segmentation_ute.py"


# You need to make the shell script executable. Open your terminal and navigate to the directory containing the run_recons.sh file, then run the following command:
# chmod +x run_recons_manual_binning.sh
# Followed by:
# ./run_recons_manual_binning.sh
