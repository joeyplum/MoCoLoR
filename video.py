# %%
import cv2
import numpy as np
import nibabel as nib

# Create a 3D matrix of images (height, width, num_frames)
filename = 'img_mocolor_10_bin_168_resolution.nii'
foldername = 'data/floret-neonatal/results/'
image_matrix = nib.load(foldername + filename)

image_matrix = np.array(image_matrix.get_fdata())
image_matrix = np.squeeze(image_matrix)

# Dimensions
num_frames = image_matrix.shape[-1]
resolution = image_matrix.shape[0]

# Select slice
slice_matrix = image_matrix[:, resolution//2, :, :]

# Normalize
min_value = np.min(slice_matrix)
max_value = np.max(slice_matrix)
slice_matrix = ((slice_matrix - min_value) /
                (max_value - min_value) * 255).astype(np.uint8)

# Set the frame rate (frames per second) for the video
frame_rate = 5

# Define the output video file name and codec
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object
out = cv2.VideoWriter(output_file, fourcc, frame_rate,
                      (resolution, resolution))

# Loop through each frame and write it to the video
for i in range(num_frames):
    frame = slice_matrix[:, :, i]
    frame_colored = cv2.cvtColor(
        frame, cv2.COLOR_GRAY2BGR)  # Convert to color format
    out.write(frame_colored)

# Release the VideoWriter and close the video file
out.release()

print("Video creation complete.")
