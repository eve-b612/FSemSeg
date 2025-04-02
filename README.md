# 🧱 FSemSeg: Building Facade Semantic Segmentation

This project performs semantic segmentation on building facades using trained deep learning models (U-Net or FCN). It predicts facade element masks and computes the **Window-to-Wall Ratio (WWR)** from the predicted segmentation.

---

## Features

- Supports **U-Net** and **Fully Convolutional Networks (FCN)**
- Computes **WWR** from predicted masks
- Handles `.jpg` and `.png` images
- Automatically downloads trained models from **Google Drive**
- Runs inference on a full test dataset folder

---

## Project Structure
FSemSeg/
├── models/
│   └── 061224_unet.keras
│   └── 061224_fcn.keras
├── test_data/
│   └── 000.png
│   └── 001.png
│   └── ...
├── predictions/
│   └── 000_unet_pred.png
│   └── 001_unet_pred.png
│   └── ...
├── scripts/
│   ├── run_inference.py
│   └── download_model.py
│   └── utils.py
├── requirements.txt
└── README.md

### 1. Clone the repository

```bash
git clone https://github.com/eve-b612/FSemSeg.git
cd FSemSeg
```
### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Run inference
Choose the model to use by editing this line in run_inference.py:
```python 
USE_UNET = True  # Set to False to use the FCN model
```
Run script
```bash
python scripts/run_inference.py
```
The script will download the appropriate model if it doesn't exist locally.

Inference will run over all .jpg and .png images in the test_data/ folder.

Output masks will be saved in predictions/.

WWR values will be printed to the terminal.