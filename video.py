# %%
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from skimage.transform import resize

matplotlib.use('TkAgg')

single_frame = False

# Create a 3D matrix of images (height, width, num_frames)
filename = 'img_nufft_24_bin_320mm_FOV_1mm_recon_resolution.nii'
foldername = '/storage/Joey/MoCoLoR/data/floret-neonatal-20230803/results/'
image_matrix = nib.load(foldername + filename)

image_matrix = np.array(image_matrix.get_fdata())
image_matrix = np.squeeze(image_matrix)

# Optional: omit first frame (if looking at specific/jacs vent image)
# image_matrix = image_matrix[..., 1:]
slice_min = 10
slice_max = image_matrix.shape[0] - slice_min
image_matrix = image_matrix[slice_min:slice_max, slice_min:slice_max, slice_min:slice_max, ...]

# Dimensions
num_frames = image_matrix.shape[-1]
resolution = image_matrix.shape[0]

# Select slice
slice_matrix = np.flip(
    np.rot90(image_matrix[:, int(resolution*0.5), :, :], k=3), axis=1)

# Upsample
desired_res = 1080
slice_matrix_upscaled = np.zeros((desired_res,desired_res,num_frames))
for i in range(num_frames):
    tmp = slice_matrix[:,:,i]
    tmp = resize(tmp, (desired_res,desired_res), mode='constant', preserve_range=True)
    slice_matrix_upscaled[:, :, i] = tmp

# Overwrite
slice_matrix = slice_matrix_upscaled

# Normalize
min_value = np.min(slice_matrix)
max_value = np.max(slice_matrix)
slice_matrix = ((slice_matrix - min_value) /
                (max_value - min_value) * 255).astype(np.uint8)
    
# Set the frame rate (frames per second) for the video
# 10 bin
frame_rate = 7

# Define the output video file name and codec
output_file = foldername + filename[:-4] + '.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object
out = cv2.VideoWriter(output_file, fourcc, frame_rate,
                      (desired_res, desired_res))

# Loop through each frame and write it to the video
for i in range(num_frames):
    frame = slice_matrix[:, :, i]
    frame_colored = cv2.cvtColor(
        frame, cv2.COLOR_GRAY2BGR)  # Convert to color format
    out.write(frame_colored)

# Release the VideoWriter and close the video file
out.release()

# Save an image
if single_frame:
    # Display the image using imshow
    plt.imshow(slice_matrix[..., 1], cmap="gray")
    plt.axis('off')  # Turn off axis
    plt.show()  # Display the plot

    # Save the displayed image
    # Change the file extension based on the desired format
    output_path = foldername + 'single_frame.png'
    plt.savefig(output_path, bbox_inches='tight',
                pad_inches=0, dpi=300)  # Save the current plot


print("Video creation complete.")

# %%
