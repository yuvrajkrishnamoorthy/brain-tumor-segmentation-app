# 🧠 Brain Tumor Segmentation App

This Streamlit app uses a trained U-Net deep learning model to perform **semantic segmentation** on brain MRI scans. It supports both standard grayscale image formats (`.png`, `.jpg`, `.tif`) and **DICOM** files. The app provides real-time tumor segmentation overlays and calculates tumor **area (mm²)** or **volume (mm³)** based on the input.

---

## 🚀 Features

- 📤 Upload single or multiple MRI images (2D formats or DICOM series)
- 🧠 U-Net-based brain tumor segmentation
- 📐 Automatic calculation of tumor area (2D) or volume (3D)
- 🖼️ High-resolution image and mask overlay visualization
- 🧾 Real-time results with an interactive Streamlit interface

---

## 📁 Project Structure

brain-tumor-segmentation-app/
├── app.py                  # Streamlit application
├── model/
│   └── unet_brain_mri.pth  # Trained U-Net model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

---

## 🧪 Tech Stack

- Python (PyTorch, Streamlit)
- DICOM support via `pydicom`
- Image handling with Pillow, NumPy, Matplotlib

---

## 💻 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/brain-tumor-segmentation-app.git
cd brain-tumor-segmentation-app

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py



⸻

📏 Tumor Measurement
	•	🖼️ 2D Images → Tumor Area in mm²
	•	🧊 3D Series (DICOM or folder of 2D slices) → Tumor Volume in mm³

⸻

👨‍💻 Author

Built by Yuvraj Krishnamoorthy
🔗 Connect on LinkedIn https://www.linkedin.com/in/yuvraj-krishnamoorthy/

⸻

📄 License

This project is licensed under the MIT License.

