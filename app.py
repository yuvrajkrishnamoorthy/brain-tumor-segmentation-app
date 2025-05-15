import streamlit as st
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
from pathlib import Path

# --------------------------
# U-Net Model
# --------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.middle = conv_block(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.middle(self.pool(e3))
        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.out(d1))

@st.cache_resource
def load_model():
    model = UNet()
    model.load_state_dict(torch.load("model/unet_brain_mri.pth", map_location="cpu"))
    model.eval()
    return model

def predict_mask(image):
    model = load_model()
    image = image.convert("L")
    image_np = np.array(image)
    orig_size = image.size
    image_resized = image.resize((128, 128))
    tensor_img = TF.to_tensor(image_resized).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor_img).squeeze().numpy()
    mask = (pred > 0.5).astype(np.uint8)
    mask_resized = Image.fromarray(mask * 255).resize(orig_size, resample=Image.NEAREST)
    return np.array(mask_resized) > 0

def display_overlay(original, mask, title="Prediction", pixel_spacing=None):
    fig, ax = plt.subplots(figsize=(original.width / 100, original.height / 100), dpi=100)
    ax.imshow(original, cmap="gray")
    ax.imshow(mask, cmap="Reds", alpha=0.4)
    ax.set_title(title)
    ax.axis("off")
    if pixel_spacing:
        ax.set_xlabel(f"Width (mm): {pixel_spacing[1]}")
        ax.set_ylabel(f"Height (mm): {pixel_spacing[0]}")
    st.pyplot(fig)

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(layout="wide")
st.title("🧠 Brain Tumor Segmentation - 2D & 3D with Area/Volume Calculation")

file_type = st.radio("Upload Type", ["Single 2D Image", "Multiple Image Files (2D/3D)"])

if file_type == "Single 2D Image":
    uploaded_file = st.file_uploader("Upload a 2D MRI image", type=["tif", "png", "jpg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        pred_mask = predict_mask(image)
        display_overlay(image, pred_mask, "Tumor Overlay")

        pixel_area_mm2 = 0.5 * 0.5  # default pixel spacing
        tumor_area = np.sum(pred_mask) * pixel_area_mm2
        st.success(f"🧼 Estimated Tumor Area: **{tumor_area:.2f} mm²**")

elif file_type == "Multiple Image Files (2D/3D)":
    image_files = st.file_uploader(
        "Upload multiple images or DICOM files", type=["dcm", "jpg", "png", "tif"], accept_multiple_files=True
    )
    if image_files:
        image_files = sorted(image_files, key=lambda x: x.name)
        tumor_volume_mm3 = 0
        slice_images = []
        pixel_spacing = [0.5, 0.5]
        slice_thickness = 1.0

        for file in image_files:
            try:
                if file.name.endswith(".dcm"):
                    dcm = pydicom.dcmread(file)
                    img = Image.fromarray(dcm.pixel_array).convert("L")
                    pixel_spacing = [float(v) for v in getattr(dcm, "PixelSpacing", [0.5, 0.5])]
                    slice_thickness = float(getattr(dcm, "SliceThickness", 1.0))
                else:
                    img = Image.open(file).convert("L")
                pred_mask = predict_mask(img)
                slice_images.append((img, pred_mask))
                tumor_volume_mm3 += np.sum(pred_mask) * pixel_spacing[0] * pixel_spacing[1] * slice_thickness
            except Exception as e:
                st.warning(f"Skipped a file due to error: {str(e)}")

        st.success(f"🧼 Estimated Tumor Volume: **{tumor_volume_mm3:.2f} mm³**")

        if slice_images:
            index = st.slider("Preview Slice", 0, len(slice_images)-1, len(slice_images)//2)
            img, mask = slice_images[index]
            display_overlay(img, mask, f"Slice {index+1} Prediction", pixel_spacing)
