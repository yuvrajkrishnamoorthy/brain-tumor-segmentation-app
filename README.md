# 🧠 Brain Tumor Segmentation App

A simple Streamlit web app that uses a trained U-Net model to segment brain tumors from MRI scans.

## 🚀 Features

- Upload MRI images (DICOM, PNG, JPG, TIF)
- Automatically segments tumor using a trained U-Net model
- Displays the tumor overlay on the original scan

## 🧪 Tech Stack

- PyTorch
- Streamlit
- pydicom
- Pillow, NumPy, Matplotlib

## 📦 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

## 📁 Project Structure

brain-tumor-segmentation-app/
├── app.py
├── model/
│   └── unet_brain_mri.pth
├── requirements.txt
└── README.md

Built by Yuvraj Krishnamoorthy  
👉 [LinkedIn](https://www.linkedin.com/in/yuvrajkrishnamoorthy)