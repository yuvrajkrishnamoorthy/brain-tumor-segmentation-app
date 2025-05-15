



# 🧠 Brain Tumor Segmentation App

This Streamlit app uses a trained U-Net model to perform semantic segmentation on brain MRI scans, predicting tumor regions.

---

## 🚀 Features

- Upload grayscale MRI images (`.tif`, `.png`, or `.jpg`)
- U-Net-based tumor segmentation
- Real-time prediction and visualization

---

## 📁 Project Structure

brain-tumor-segmentation-app/
├── app.py                  # Streamlit app
├── model/
│   └── unet_brain_mri.pth  # Trained U-Net model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

---

## 💻 How to Run

```bash
# Clone the repository
git clone https://github.com/yuvrajkrishnamoorthy/brain-tumor-segmentation-app.git
cd brain-tumor-segmentation-app

# Create and activate a virtual environment (optional)
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py



⸻

📸 Demo

Upload a .tif image from the dataset and view predicted tumor regions overlaid on the original scan in real-time.

⸻

🧠 Model Info
	•	Architecture: U-Net
	•	Trained on: LGG Brain MRI Segmentation Dataset (Kaggle)

⸻

🙋‍♂️ Author

Built by Yuvraj Krishnamoorthy
🔗 LinkedIn

⸻

📄 License

This project is licensed under the MIT License.

---

Let me know if you'd like to add screenshots, a model training section, or how to support DICOM uploads too.# 🧠 Brain Tumor Segmentation App

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

📄 License

This project is licensed under the MIT License.

