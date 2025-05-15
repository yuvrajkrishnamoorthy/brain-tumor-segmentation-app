import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pydicom

# --- Model Definition ---
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

# --- Load Model ---
model = UNet()
model.load_state_dict(torch.load("model/unet_brain_mri.pth", map_location="cpu"))
model.eval()

# --- Streamlit App ---
st.title("🧠 Brain Tumor Segmentation")

uploaded_file = st.file_uploader("Upload an MRI (DICOM, PNG, JPG, TIF)", type=["dcm", "png", "jpg", "tif"])

if uploaded_file:
    if uploaded_file.name.endswith(".dcm"):
        dcm = pydicom.dcmread(uploaded_file)
        image_np = dcm.pixel_array.astype(np.float32)
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
        image = Image.fromarray((image_np * 255).astype(np.uint8)).convert("L").resize((128, 128))
    else:
        image = Image.open(uploaded_file).convert("L").resize((128, 128))

    st.image(image, caption="Original MRI", use_column_width=True)

    # Inference
    input_tensor = TF.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        pred = model(input_tensor)
        pred_mask = (pred.squeeze() > 0.5).numpy()

    # Overlay
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.imshow(pred_mask, cmap="Reds", alpha=0.4)
    ax.axis("off")
    st.pyplot(fig)