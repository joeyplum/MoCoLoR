from sigpy_e.linop_e import NFTs, Diags, DLD, Vstacks
import sigpy_e.cfl as cfl
import nibabel as nib
import logging
import os
import sigpy.mri as mr
import sigpy_e.reg as reg
import sigpy_e.ext as ext
import argparse
import sigpy as sp
import scipy.ndimage as ndimage_c
import numpy as np

import sys
sys.path.append("./sigpy_e/")

# Usage
# python recon_xdgrasp_xe_diffusion.py "data/spiral-803H-001/" --scan_res 0.5 --lambda_TV 5e-2


# IO parameters
parser = argparse.ArgumentParser(description='XD-GRASP recon.')

parser.add_argument('--res_scale', type=float, default=1,
                    help='scale of resolution, full res == 1')
parser.add_argument('--scan_res', type=float, default=60,
                    help='scan matrix size')
parser.add_argument('--recon_res', type=float, default=60,
                    help='recon matrix size')
parser.add_argument('--fov_x', type=float, default=1,
                    help='scale of FOV x, full res == 1')
parser.add_argument('--fov_y', type=float, default=1,
                    help='scale of FOV y, full res == 1')
parser.add_argument('--fov_z', type=float, default=1,
                    help='scale of FOV z, full res == 1')

parser.add_argument('--lambda_TV', type=float, default=5e-2,
                    help='TV regularization, default is 0.05')
parser.add_argument('--outer_iter', type=int, default=25,
                    help='Num of Iterations.')

parser.add_argument('--vent_flag', type=int, default=0,
                    help='output jacobian determinant and specific ventilation')
parser.add_argument('--n_ref_vent', type=int, default=0,
                    help='reference frame for ventilation')

parser.add_argument('--device', type=int, default=0,
                    help='Computing device.')

parser.add_argument('fname', type=str,
                    help='Prefix of raw data and output(_mrL).')
args = parser.parse_args()

#
res_scale = args.res_scale
scan_resolution = args.scan_res
recon_resolution = args.recon_res
fname = args.fname
lambda_TV = args.lambda_TV
device = args.device
outer_iter = args.outer_iter
fov_scale = (args.fov_x, args.fov_y, args.fov_z)
vent_flag = args.vent_flag
n_ref_vent = args.n_ref_vent

print('Reconstruction started.')

# data loading
data = np.load(os.path.join(fname, 'bksp.npy'))
traj = np.real(np.load(os.path.join(fname, 'bcoord.npy')))
dcf = np.sqrt(np.load(os.path.join(fname, 'bdcf.npy')))

nf_scale = res_scale
nf_arr = np.sqrt(np.sum(traj[0, 0, :, :]**2, axis=1))
nf_e = np.sum(nf_arr < np.max(nf_arr)*nf_scale)
# scale = fov_scale
scale = (scan_resolution, scan_resolution)  # Added JWP
traj[..., 0] = traj[..., 0]*scale[0]
traj[..., 1] = traj[..., 1]*scale[1]

traj = traj[..., :nf_e, :]
data = data[..., :nf_e]
data = data[:, :, 5, :, :]  # Select middle slice
dcf = dcf[..., :nf_e]
dcf = dcf[:, 0, :, :]  # Remove coil dimension

nphase, nCoil, npe, nfe = data.shape
# tshape = (np.int(np.max(traj[..., 0])-np.min(traj[..., 0])), np.int(np.max(
#     traj[..., 1])-np.min(traj[..., 1])), np.int(np.max(traj[..., 2])-np.min(traj[..., 2])))
# Or use input settings
# tshape = (int(args.fov_x), int(args.fov_y), int(args.fov_z))
tshape = (int(recon_resolution), int(recon_resolution))

# calibration
ksp = np.reshape(np.transpose(data, (1, 0, 2, 3)), (nCoil, nphase*npe, nfe))
dcf2 = np.reshape(dcf**2, (nphase*npe, nfe))
coord = np.reshape(traj, (nphase*npe, nfe, 2))

mps = np.ones((1,) + (tshape))
S = sp.linop.Multiply(tshape, mps)

# recon
PFTSs = []
for i in range(nphase):
    FTs = NFTs((nCoil,)+tshape, traj[i, ...], device=sp.Device(device))
    W = sp.linop.Multiply((nCoil, npe, nfe,), dcf[i, :, :])
    FTSs = W*FTs*S
    PFTSs.append(FTSs)
PFTSs = Diags(PFTSs, oshape=(nphase, nCoil, npe, nfe,),
              ishape=(nphase,)+tshape)

# preconditioner
wdata = data*dcf[:, np.newaxis, :, :]
tmp = PFTSs.H*PFTSs*np.complex64(np.ones((nphase,)+tshape))
L = np.mean(np.abs(tmp))


# reconstruction
q2 = np.zeros((nphase,)+tshape, dtype=np.complex64)
Y = np.zeros_like(wdata)
q20 = np.zeros_like(q2)
res_norm = np.zeros((outer_iter, 1))

logging.basicConfig(level=logging.INFO)

sigma = 0.4
tau = 0.4
for i in range(outer_iter):
    Y = (Y + sigma*(1/L*PFTSs*q2-wdata))/(1+sigma)

    q20 = q2
    q2 = np.complex64(ext.TVt_prox(q2-tau*PFTSs.H*Y, lambda_TV))
    res_norm[i] = np.linalg.norm(q2-q20)/np.linalg.norm(q2)
    logging.info('outer iter:{}, res:{}'.format(i, res_norm[i]))

    np.save(os.path.join(fname, 'prL.npy'), q2)

# recon using standard inverse nufft
print(str(data.shape))
print(str(traj.shape))
print(str(dcf.shape))
F = sp.linop.NUFFT(mps.shape,
                   coord=traj[0, :, :, :],
                   oversamp=1.25,
                   width=4,
                   toeplitz=True)
D = sp.linop.Multiply(F.oshape, dcf[0, :, :])  # Optional
A = D * F * S
LL = sp.app.MaxEig(A.N, dtype=np.complex64,
                   device=sp.Device(device)).run() * 1.01
A = np.sqrt(1/LL) * A
img_nufft = np.zeros((nphase, recon_resolution, recon_resolution))
for i in range(nphase):
    b_dcf = np.reshape((data[i, 0, :, :] * dcf[0, :, :]),
                       ((1,) + data[i, 0, :, :].shape))
    print(str(b_dcf.shape))
    img_nufft[i, :, :] = abs(A.H * b_dcf)

# Calculate ADC

# Initialize settings
ADC_map_nufft = np.zeros((recon_resolution, recon_resolution))
ADC_map_cs = np.zeros((recon_resolution, recon_resolution))
b_values = np.array([0, 10, 20, 30])

# Iterate through image data to calculate ADC
for i in range(recon_resolution):
    for j in range(recon_resolution):
        signal = abs(img_nufft[:, i, j])
        log_signal = np.log(signal)
        model = np.polyfit(b_values, log_signal, 1)
        slope = model[0]
        ADC_map_nufft[i, j] = - slope
        signal = abs(q2[:, i, j])
        log_signal = np.log(signal)
        model = np.polyfit(b_values, log_signal, 1)
        slope = model[0]
        ADC_map_cs[i, j] = - slope

# Cleanup
ADC_map_nufft[ADC_map_nufft < 0] = 0
ADC_map_nufft[ADC_map_nufft > 0.14] = 0
ADC_map_cs[ADC_map_cs < 0] = 0
ADC_map_cs[ADC_map_cs > 0.14] = 0


# Check whether a specified save data path exists
results_exist = os.path.exists(fname + "/results")

# Create a new directory because the results path does not exist
if not results_exist:
    os.makedirs(fname + "/results")
    print("A new directory inside: " + fname +
          " called 'results' has been created.")

# Save images as Nifti files
# Build an array using matrix multiplication
scaling_affine = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

# Rotate gamma radians about axis i
cos_gamma = np.cos(0)
sin_gamma = np.sin(0)
rotation_affine_1 = np.array([[1, 0, 0, 0],
                              [0, cos_gamma, -sin_gamma,  0],
                              [0, sin_gamma, cos_gamma, 0],
                              [0, 0, 0, 1]])
cos_gamma = np.cos(np.pi)
sin_gamma = np.sin(np.pi)
rotation_affine_2 = np.array([[cos_gamma, 0, sin_gamma, 0],
                              [0, 1, 0, 0],
                              [-sin_gamma, 0, cos_gamma, 0],
                              [0, 0, 0, 1]])
cos_gamma = np.cos(0)
sin_gamma = np.sin(0)
rotation_affine_3 = np.array([[cos_gamma, -sin_gamma, 0, 0],
                              [sin_gamma, cos_gamma, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
rotation_affine = rotation_affine_1.dot(
    rotation_affine_2.dot(rotation_affine_3))

# Apply translation
translation_affine = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

# Multiply matrices together
aff = translation_affine.dot(rotation_affine.dot(scaling_affine))

ni_img = nib.Nifti1Image(abs(np.moveaxis(q2, 0, -1)), affine=aff)
nib.save(ni_img, fname + '/results/img_xdgrasp_' + str(nphase) + '_bin')

ni_img = nib.Nifti1Image(abs(np.moveaxis(img_nufft, 0, -1)), affine=aff)
nib.save(ni_img, fname + '/results/img_nufft_' + str(nphase) + '_bin')

ni_img = nib.Nifti1Image(abs(np.moveaxis(ADC_map_cs, 0, -1)), affine=aff)
nib.save(ni_img, fname + '/results/ADC_xdgrasp_' + str(nphase) + '_bin')

ni_img = nib.Nifti1Image(abs(np.moveaxis(ADC_map_nufft, 0, -1)), affine=aff)
nib.save(ni_img, fname + '/results/ADC_nufft_' + str(nphase) + '_bin')
