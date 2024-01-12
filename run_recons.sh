#!/bin/bash

# Replace 'python' with the appropriate command if 'python' doesn't work in your environment.
PYTHON_EXECUTABLE=python

# Function to run the binning_quantile.py script with given arguments
run_binning_quantile() {
    $PYTHON_EXECUTABLE binning_hilbert_dynamic.py --fname data/floret-740H-032c/ --nbins 10 --plot 1 --nprojections 10000
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_xdgrasp1() {
    $PYTHON_EXECUTABLE recon_xdgrasp_npy.py data/floret-740H-032c/ --lambda_TV 0.05 --vent_flag 1 --recon_res 220 --scan_res 220
}

# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_mocolor1() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py data/floret-740H-032c/ --lambda_lr 0.05 --vent_flag 1 --recon_res 220 --scan_res 220 --mr_cflag 1
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_nufft() {
    $PYTHON_EXECUTABLE recon_dcf_nufft_npy.py data/floret-740H-032c/ --lambda_TV 0 --vent_flag 0 --recon_res 220 --scan_res 220
}

echo "Running binning_quantile.py ..."
run_binning_quantile
echo "Finished binning_quantile.py"

echo "Running recon_dcf_nufft_npy.py ..."
# run_recon_nufft
echo "Finished recon_dcf_nufft_npy.py"

echo "Running recon_xdgrasp_npy.py ..."
# run_recon_xdgrasp1
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_mocolor_npy.py ..."
run_recon_mocolor1
echo "Finished recon_mocolor_npy.py"


# You need to make the shell script executable. Open your terminal and navigate to the directory containing the run_recons.sh file, then run the following command:
# chmod +x run_recons.sh
# Followed by:
# ./run_recons.sh



