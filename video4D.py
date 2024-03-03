# %%
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

single_frame = False

# Create a 3D matrix of images (height, width, num_frames)
filename = 'img_mocolor_32_bin_480mm_FOV_3mm_recon_resolution.nii'
foldername = '/storage/Joey/MoCoLoR/data/floret-ch-001v2/results/'
image_matrix = nib.load(foldername + filename)

image_matrix = np.array(image_matrix.get_fdata())
image_matrix = np.squeeze(image_matrix)

# Optional: omit first frame (if looking at specific/jacs vent image)
image_matrix = image_matrix[..., 1:]
slice_min = 20
slice_max = image_matrix.shape[0] - slice_min
image_matrix = image_matrix[slice_min:slice_max, slice_min:slice_max, slice_min:slice_max, ...]

# Optional: only include the middle X% of slices
image_matrix = image_matrix[:, int(
    0.3*image_matrix.shape[0]):int(0.7*image_matrix.shape[0]), ...]

# Dimensions
resolution = image_matrix.shape[0]
num_slices = image_matrix.shape[1]
num_frames = image_matrix.shape[-1]

# Set the frame rate (frames per second) for the video
# 10 bin
frame_rate = 15

# Define the output video file name and codec
output_file = foldername + filename[:-4] + '.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object
out = cv2.VideoWriter(output_file, fourcc, frame_rate,
                      (resolution, resolution))

# Make an array that goes up and then back down
list_iter = np.concatenate(
    [np.arange(num_slices - 1), np.arange(num_slices-1, -1, -1)])
num_iter = len(list_iter)

# Loop through each frame and write it to the video
# for i in range(num_slices):
for i in range(num_iter):
    slice_number = list_iter[i]
    frame_number = list_iter[i] % num_frames

    # Select slice
    slice_matrix = np.flip(
        np.rot90(image_matrix[:, slice_number, :, :], k=3), axis=1)

    # Normalize
    min_value = np.min(slice_matrix)
    max_value = np.max(slice_matrix)
    slice_matrix = ((slice_matrix - min_value) /
                    (max_value - min_value) * 255).astype(np.uint8)

    frame = slice_matrix[:, :, frame_number]
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
