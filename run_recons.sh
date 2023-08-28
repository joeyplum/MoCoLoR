#!/bin/bash

# Replace 'python' with the appropriate command if 'python' doesn't work in your environment.
PYTHON_EXECUTABLE=python

# Function to run the binning_quantile.py script with given arguments
run_binning_quantile() {
    $PYTHON_EXECUTABLE binning_quantile.py --fname data/floret-740H-054/ --nbins 6 --plot 0
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_xdgrasp1() {
    $PYTHON_EXECUTABLE recon_xdgrasp_npy.py data/floret-740H-054/ --lambda_TV 0.05 --vent_flag 1 --recon_res 200 --scan_res 200
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_xdgrasp2() {
    $PYTHON_EXECUTABLE recon_xdgrasp_npy.py data/floret-740H-054/ --lambda_TV 0.05 --vent_flag 1 --recon_res 120 --scan_res 200
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_xdgrasp3() {
    $PYTHON_EXECUTABLE recon_xdgrasp_npy.py data/floret-740H-054/ --lambda_TV 0.05 --vent_flag 1 --recon_res 90 --scan_res 200
}

# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_mocolor1() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py data/floret-740H-054/ --lambda_lr 0.01 --vent_flag 1 --recon_res 200 --scan_res 200 --mr_cflag 1
}

# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_mocolor2() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py data/floret-740H-054/ --lambda_lr 0.01 --vent_flag 1 --recon_res 120 --scan_res 200 --mr_cflag 1
}

# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_mocolor3() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py data/floret-740H-054/ --lambda_lr 0.01 --vent_flag 1 --recon_res 90 --scan_res 200 --mr_cflag 1
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_nufft() {
    $PYTHON_EXECUTABLE recon_dcf_nufft_npy.py data/floret-740H-054/ --lambda_TV 0 --vent_flag 1 --recon_res 200 --scan_res 200
}

echo "Running binning_quantile.py ..."
run_binning_quantile
echo "Finished binning_quantile.py"

echo "Running recon_xdgrasp_npy.py ..."
run_recon_xdgrasp1
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_xdgrasp_npy.py ..."
# run_recon_xdgrasp2
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_xdgrasp_npy.py ..."
# run_recon_xdgrasp3
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_mocolor_npy.py ..."
run_recon_mocolor1
echo "Finished recon_mocolor_npy.py"

echo "Running recon_mocolor_npy.py ..."
# run_recon_mocolor2
echo "Finished recon_mocolor_npy.py"

echo "Running recon_mocolor_npy.py ..."
# run_recon_mocolor3
echo "Finished recon_mocolor_npy.py"

echo "Running recon_dcf_nufft_npy.py ..."
run_recon_nufft
echo "Finished recon_dcf_nufft_npy.py"

# You need to make the shell script executable. Open your terminal and navigate to the directory containing the run_recons.sh file, then run the following command:
# chmod +x run_recons.sh
# Followed by:
# ./run_recons.sh

# Function to run the binning_quantile.py script with given arguments
run_binning_quantile() {
    $PYTHON_EXECUTABLE binning_quantile.py --fname data/floret-740H-054/ --nbins 10 --plot 0
}

echo "Running binning_quantile.py ..."
run_binning_quantile
echo "Finished binning_quantile.py"

echo "Running recon_xdgrasp_npy.py ..."
run_recon_xdgrasp1
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_xdgrasp_npy.py ..."
# run_recon_xdgrasp2
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_xdgrasp_npy.py ..."
# run_recon_xdgrasp3
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_mocolor_npy.py ..."
run_recon_mocolor1
echo "Finished recon_mocolor_npy.py"

echo "Running recon_mocolor_npy.py ..."
# run_recon_mocolor2
echo "Finished recon_mocolor_npy.py"

echo "Running recon_mocolor_npy.py ..."
# run_recon_mocolor3
echo "Finished recon_mocolor_npy.py"

echo "Running recon_dcf_nufft_npy.py ..."
run_recon_nufft
echo "Finished recon_dcf_nufft_npy.py"


# Function to run the binning_quantile.py script with given arguments
run_binning_quantile() {
    $PYTHON_EXECUTABLE binning_quantile.py --fname data/floret-740H-054/ --nbins 14 --plot 0
}


echo "Running binning_quantile.py ..."
run_binning_quantile
echo "Finished binning_quantile.py"

echo "Running recon_xdgrasp_npy.py ..."
run_recon_xdgrasp1
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_xdgrasp_npy.py ..."
# run_recon_xdgrasp2
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_xdgrasp_npy.py ..."
# run_recon_xdgrasp3
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_mocolor_npy.py ..."
run_recon_mocolor1
echo "Finished recon_mocolor_npy.py"

echo "Running recon_mocolor_npy.py ..."
# run_recon_mocolor2
echo "Finished recon_mocolor_npy.py"

echo "Running recon_mocolor_npy.py ..."
# run_recon_mocolor3
echo "Finished recon_mocolor_npy.py"

echo "Running recon_dcf_nufft_npy.py ..."
run_recon_nufft
echo "Finished recon_dcf_nufft_npy.py"

