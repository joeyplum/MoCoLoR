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
import time

import sys
sys.path.append("./sigpy_e/")

# Usage
# python recon_xdgrasp_npy.py data/floret-740H-053/ --lambda_TV 0.05 --vent_flag 1 --recon_res 220 --scan_res 220

# IO parameters
parser = argparse.ArgumentParser(description='XD-GRASP recon.')

parser.add_argument('--res_scale', type=float, default=.75,
                    help='scale of resolution, full res == .75')
parser.add_argument('--scan_res', type=float, default=200,
                    help='scan matrix size')
parser.add_argument('--recon_res', type=float, default=200,
                    help='econ matrix size')

parser.add_argument('--fov_x', type=float, default=1,
                    help='scale of FOV x, full res == 1')
parser.add_argument('--fov_y', type=float, default=1,
                    help='scale of FOV y, full res == 1')
parser.add_argument('--fov_z', type=float, default=1,
                    help='scale of FOV z, full res == 1')

parser.add_argument('--lambda_TV', type=float, default=0,
                    help='TV regularization, default is 0.05')
parser.add_argument('--outer_iter', type=int, default=1,
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
tic_total = time.perf_counter()

# data loading
data = np.load(os.path.join(fname, 'bksp.npy'))
traj = np.real(np.load(os.path.join(fname, 'bcoord.npy')))
dcf = np.sqrt(np.load(os.path.join(fname, 'bdcf.npy')))

nf_scale = res_scale
nf_arr = np.sqrt(np.sum(traj[0, 0, :, :]**2, axis=1))
nf_e = np.sum(nf_arr < np.max(nf_arr)*nf_scale)
scale = (scan_resolution, scan_resolution, scan_resolution)  # Added JWP
traj[..., 0] = traj[..., 0]*scale[0]
traj[..., 1] = traj[..., 1]*scale[1]
traj[..., 2] = traj[..., 2]*scale[2]

# Optional: undersample along freq encoding - JWP 20230815
# traj = traj[..., :nf_e, :]
# data = data[..., :nf_e]
# dcf = dcf[..., :nf_e]

nphase, nCoil, npe, nfe = data.shape
tshape = (np.int(np.max(traj[..., 0])-np.min(traj[..., 0])), np.int(np.max(
    traj[..., 1])-np.min(traj[..., 1])), np.int(np.max(traj[..., 2])-np.min(traj[..., 2])))
# Or use manual input settings
tshape = (int(recon_resolution), int(
    recon_resolution), int(recon_resolution))

print('Number of phases used in this reconstruction: ' + str(nphase))
print('Number of coils: ' + str(nCoil))
print('Number of phase encodes: ' + str(npe))
print('Number of frequency encodes: ' + str(nfe))

# calibration
ksp = np.reshape(np.transpose(data, (1, 0, 2, 3)), (nCoil, nphase*npe, nfe))
dcf2 = np.reshape(dcf**2, (nphase*npe, nfe))
coord = np.reshape(traj, (nphase*npe, nfe, 3))

mps = ext.jsens_calib(ksp, coord, dcf2, device=sp.Device(0), ishape=tshape)
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
    tic = time.perf_counter()
    Y = (Y + sigma*(1/L*PFTSs*q2-wdata))/(1+sigma)

    q20 = q2
    q2 = np.complex64(ext.TVt_prox(q2-tau*PFTSs.H*Y, lambda_TV))
    res_norm[i] = np.linalg.norm(q2-q20)/np.linalg.norm(q2)
    toc = time.perf_counter()
    logging.info(' outer iter:{}, res:{}, {}sec'.format(
        i, res_norm[i], int(toc - tic)))

    # np.save(os.path.join(fname, 'prL.npy'), q2)
    # np.save(os.path.join(fname, 'prL_residual_{}.npy'.format(lambda_TV)), res_norm)
# q2 = np.load(os.path.join(fname, 'prL.npy'))
# jacobian determinant & specific ventilation
if vent_flag == 1:
    tic = time.perf_counter()
    print('Jacobian Determinant and Specific Ventilation...')
    jacs = []
    svs = []
    q2 = np.abs(np.squeeze(q2))
    q2 = q2/np.max(q2)
    for i in range(nphase):
        jac, sv = reg.ANTsJac(np.abs(q2[n_ref_vent]), np.abs(q2[i]))
        jacs.append(jac)
        svs.append(sv)
    jacs = np.asarray(jacs)
    svs = np.asarray(svs)
    # np.save(os.path.join(fname, 'jac_nufft.npy'), jacs)
    # np.save(os.path.join(fname, 'sv_nufft.npy'), svs)
    toc = time.perf_counter()
    print('time elapsed for ventilation metrics: {}sec'.format(int(toc - tic)))

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
    nib.save(ni_img, fname + '/results/img_nufft_' + str(nphase) +
             '_bin_' + str(int(recon_resolution)) + '_resolution')

    if vent_flag == 1:
        ni_img = nib.Nifti1Image(np.moveaxis(svs, 0, -1), affine=aff)
        nib.save(ni_img, fname + '/results/sv_nufft_' +
                 str(nphase) + '_bin_' + str(int(recon_resolution)) + '_resolution')

        ni_img = nib.Nifti1Image(np.moveaxis(jacs, 0, -1), affine=aff)
        nib.save(ni_img, fname + '/results/jacs_nufft_' + str(nphase) +
                 '_bin_' + str(int(recon_resolution)) + '_resolution')

    toc_total = time.perf_counter()
    print('total time elapsed: {}mins'.format(int(toc_total - tic_total)/60))
