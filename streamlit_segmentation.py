import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
import tempfile
import torch
import torch.nn.functional as F
from networks.net_factory import net_factory  # Assuming this is your model loader

# Load the pre-trained model
@st.cache_resource
def load_model(model_path):
    net = net_factory(net_type="unet", in_chns=1, class_num=4)  # Adjust based on your architecture
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()
    return net

# Predict segmentation
# Predict segmentation
def predict_segmentation(model, image):
    """
    Perform segmentation on a 3D volume by processing each slice independently.
    """
    depth = image.shape[2]  # Number of slices in the depth dimension
    segmented_volume = []

    for i in range(depth):
        # Extract a single 2D slice
        image_slice = image[:, :, i]

        # Prepare the input tensor for the model
        image_tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, H, W]

        # Perform prediction
        with torch.no_grad():
            output = model(image_tensor)  # Model expects 4D input: [batch_size, channels, height, width]
            pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).numpy()  # Shape: [H, W]

        # Store the segmented slice
        segmented_volume.append(pred)

    # Stack all the segmented slices to form a 3D volume
    return np.stack(segmented_volume, axis=-1)


# Upload file
uploaded_file = st.file_uploader("Upload an image file (.nii.gz)", type=["nii.gz"])

if uploaded_file:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load the image using nibabel
    img = nib.load(tmp_path).get_fdata()
    os.remove(tmp_path)  # Clean up the temporary file

    # Debugging: Print the shape of the image
    st.write(f"Image shape: {img.shape}")

    # Display original image
    st.subheader("Uploaded Image")
    if len(img.shape) == 3:
        # Handle 3D images
        slice_idx = st.slider("Select Slice", 0, img.shape[2] - 1, img.shape[2] // 2)
        img_slice = img[:, :, slice_idx]
    elif len(img.shape) == 4:
        # Handle 4D images (e.g., multi-channel data)
        st.error("4D images are not supported in this visualization tool.")
    else:
        # Unsupported shape
        st.error("Unsupported image dimensions. Ensure the uploaded file is 3D.")
        st.stop()

    plt.figure(figsize=(5, 5))
    plt.imshow(img_slice, cmap="gray")
    plt.axis("off")
    st.pyplot(plt)

    # Load the model
    model_path = r"C:\Users\rajwn\Videos\Manus_Malaka\3DCardiac\model\supervised\ACDC_BCP_7_labeled\self_train\unet_best_model.pth"
    model = load_model(model_path)

    # Perform segmentation
    st.subheader("Segmentation Result")
    segmented = predict_segmentation(model, img)

    # Display segmented result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Slice")
    plt.imshow(img_slice, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmented Slice")
    plt.imshow(img_slice, cmap="gray")
    plt.imshow(segmented[:, :, slice_idx], cmap="jet", alpha=0.5)
    plt.axis("off")
    st.pyplot(plt)

    st.success("Segmentation completed!")
