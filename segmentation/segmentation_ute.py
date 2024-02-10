import time
import argparse
from skimage.transform import resize
import nibabel as nib
from keras.models import load_model
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

model_name = 'model.200.hdf5'
madel_path = 'segmentation/'
model = load_model(madel_path+model_name, compile=False)

if __name__ == "__main__":
    # IO parameters
    parser = argparse.ArgumentParser(
        description='UTE segmentation tool.')

    parser.add_argument('--fname', type=str,
                        help='folder name (e.g. data/floret-neonatal/).')
    parser.add_argument('--filename', type=str,
                        help='file name (e.g. img_mocolor_10_bin_150_res.nii')
    parser.add_argument('--plot', type=int, default=1,
                        help='show plots of waveforms, 1=True or 0=False.')
    parser.add_argument('--mask', type=int, default=1,
                        help='masks to include. 0=all, 1=lungs only, 2=body and lungs only.')

    args = parser.parse_args()

    folder = args.fname
    file = args.filename
    show_plot = args.plot
    mask = args.mask

    tic_total = time.perf_counter()

    # Prediction
    def AutoSeg_UTE(Image):# Require 3D input
        ndim = np.ndim(Image)
        Im_shape = Image.shape
        predicted_mask = np.zeros_like((Image))
        seg_vec = np.zeros(Image.shape[-1])
        for i in range(Image.shape[-1]):
            img_input = Image[:,:,i]
            img_input = resize(img_input, (256,256), mode='constant', preserve_range=True)
            img_input = (img_input - np.min(img_input)) / (np.max(img_input) - np.min(img_input))  
            img_input = np.expand_dims(img_input, 0)
            img_input = np.expand_dims(img_input, -1)
            prediction = (model.predict(img_input))
            predicted_img = np.argmax(prediction, axis=3)[0,:,:]
            predicted_img = resize(predicted_img, (Im_shape[0],Im_shape[1]), mode='constant', preserve_range=True)
            predicted_img = np.round(predicted_img).astype(int)
            predicted_mask[:,:,i] = predicted_img
            # Check if the slice has values 2 or 3
            if np.any((predicted_img == 2) | (predicted_img == 3)):
                seg_vec[i] = 1
                        
        # Find the indices of ones in the vector
        one_indices = [i for i, val in enumerate(seg_vec) if val == 1]

        # Find the consecutive sequences of ones
        consecutive_sequences = []
        current_sequence = []

        for i in one_indices:
            if current_sequence and i != current_sequence[-1] + 1:
                consecutive_sequences.append(current_sequence)
                current_sequence = []
            current_sequence.append(i)

        # Check the last sequence
        if current_sequence:
            consecutive_sequences.append(current_sequence)

        # Filter and keep only the long sequences
        min_sequence_length = 40  # Adjust this threshold as needed
        filtered_sequences = [sequence for sequence in consecutive_sequences if len(sequence) >= min_sequence_length]

        # Create a new vector with zeros and ones based on the filtered sequences
        result_vector = [1 if i in sequence else 0 for i in range(len(seg_vec)) for sequence in filtered_sequences]
        for i in range(Image.shape[-1]):
            if result_vector[i] == 0:
                # Check if the slice has values 2 or 3
                if np.any((predicted_mask[:, :, i] == 2) | (predicted_mask[:, :, i] == 3)):
                    # Set all pixels with values 2 or 3 to 0
                    predicted_mask[:, :, i][np.isin(predicted_mask[:, :, i], [2, 3])] = 0
        
    
        return predicted_mask

    # Load
    image_file = folder+file
    img_test = nib.load(image_file).get_fdata()                                                                                                                                                       
    img_test = np.rot90(img_test)
    img_test = np.flip(img_test, 0)

    if show_plot == 1:
        test_slice = img_test.shape[1]//2
        test_bin = img_test.shape[-1]//2
        plt.figure()
        plt.title('Sample slice')
        plt.imshow(img_test[:, :, test_slice, test_bin], cmap='gray')
        plt.show()

    # Loop through respiratory phases
    img_size = img_test.shape[0]
    N_phases = img_test.shape[-1]
    predicted_mask = np.zeros_like(img_test)

    for resp_phase in range(N_phases):
        img_tmp = img_test[:, :, :, resp_phase]
        predicted_mask[..., resp_phase] = AutoSeg_UTE(img_tmp)

    if show_plot == 1:
        test_slice = img_test.shape[1]//2
        plt.figure(figsize=(6, 9))
        plt.subplot(211)
        plt.title('Sample slice')
        plt.imshow(img_tmp[:, :, test_slice], cmap='gray')
        plt.subplot(212)
        plt.title('Prediction (final respiratory phase only)')
        plt.imshow(predicted_mask[:, :, test_slice, -1],
                   cmap='jet', vmin=0, vmax=4)
        plt.show()

    if mask == 0:
        print(
            "Mask exported with: background = 0, body = 1, left lung = 2, right lung = 4.")
    elif mask == 1:
        print("Mask exported with: background = 0, lung = 1.")
        predicted_mask = (predicted_mask > 1).astype('uint8')
    else:
        print("Mask exported with: background = 0, lung and body = 1.")
        predicted_mask = (predicted_mask > 0).astype('uint8')

    # Reverse the transformations for saving the mask again
    mask_tmp = np.flip(predicted_mask, 0)
    mask_tmp = np.rot90(mask_tmp, -1)

    # Save
    ni_img = nib.Nifti1Image(
        abs(mask_tmp), affine=np.eye(4))
    nib.save(ni_img, folder + "mask_" + file[4:])
    toc_total = time.perf_counter()
    print('total segmentation time elapsed: {}mins'.format(
        int(toc_total - tic_total)/60))
