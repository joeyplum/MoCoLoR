#!/bin/bash

# Replace 'python' with the appropriate command if 'python' doesn't work in your environment.
PYTHON_EXECUTABLE=python

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_xdgrasp() {
    $PYTHON_EXECUTABLE recon_xdgrasp_npy.py data/floret-740H-053/ --lambda_TV 0.05 --vent_flag 1 --recon_res 220 --scan_res 220
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_mo() {
    $PYTHON_EXECUTABLE recon_xdgrasp_npy.py data/floret-740H-053/ --lambda_TV 0.05 --vent_flag 1 --recon_res 110 --scan_res 220
}

# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_color() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py data/floret-740H-053/ --lambda_lr 0.05 --vent_flag 1 --recon_res 220 --scan_res 220 --mr_cflag 1
}

# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_color() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py data/floret-740H-053/ --lambda_lr 0.05 --vent_flag 1 --recon_res 110 --scan_res 220 --mr_cflag 1
}

echo "Running recon_xdgrasp_npy.py ..."
run_recon_xdgrasp
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_xdgrasp_npy.py ..."
run_recon_xdgrasp
echo "Finished recon_xdgrasp_npy.py"

echo "Running recon_mo_npy.py ..."
run_recon_mocolor
echo "Finished recon_mo_npy.py"

echo "Running recon_mo_npy.py ..."
run_recon_mocolor
echo "Finished recon_mo_npy.py"

# Now, you need to make the shell script executable. Open your terminal and navigate to the directory containing the run_recons.sh file, then run the following command:
# chmod +x run_recons.sh
# Followed by:
# ./run_recons.sh

