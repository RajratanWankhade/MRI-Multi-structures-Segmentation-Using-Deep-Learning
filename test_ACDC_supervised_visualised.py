import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths to the files
img_path = r"C:\Users\rajwn\Videos\Manus_Malaka\3DCardiac\model\BCP\ACDC_BCP_7_labeled\unet_predictions\patient001_frame01_img.nii.gz"
gt_path = r"C:\Users\rajwn\Videos\Manus_Malaka\3DCardiac\model\BCP\ACDC_BCP_7_labeled\unet_predictions\patient001_frame01_gt.nii.gz"
pred_path = r"C:\Users\rajwn\Videos\Manus_Malaka\3DCardiac\model\BCP\ACDC_BCP_7_labeled\unet_predictions\patient001_frame01_pred.nii.gz"


# Load the data
img = nib.load(img_path).get_fdata()
gt = nib.load(gt_path).get_fdata()
pred = nib.load(pred_path).get_fdata()

# Choose a slice index to visualize
slice_idx = 10  # Change this to explore different slices
img_slice = img[slice_idx, :, :]
gt_slice = gt[slice_idx, :, :]
pred_slice = pred[slice_idx, :, :]

# Visualize the original image, ground truth, and prediction
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img_slice, cmap="gray")

# Ground Truth
plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(img_slice, cmap="gray")
plt.imshow(gt_slice, cmap="Blues", alpha=0.5)

# Prediction
plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(img_slice, cmap="gray")
plt.imshow(pred_slice, cmap="Reds", alpha=0.5)

plt.show()
