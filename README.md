# Comic Colorization Project

A deep learning project to colorize grayscale comic images using a U-Net architecture.

---

## Project Structure
```
├── API Key here/       # API key. You can borrow mine
├── Data/               # Data directory
│   ├── comic_dataset/  # Raw comic images
│   └── prepared_data/  # Preprocessed numpy arrays
├── Src/                # the model architecture
└── Notebooks/          # Jupyter notebooks (both loading data and training)

```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Sebastianrix/CNN_U-Net_Project
cd CNN_U-Net_Project
```

### 2. Create and Activate Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## GPU Setup

To enable GPU training with TensorFlow:

### Requirements:
- CUDA 11.8
- cuDNN 8.6

### Steps (Windows):
1. Install CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Download cuDNN 8.6: https://developer.nvidia.com/rdp/cudnn-download
3. Copy the contents of cuDNN's `bin/`, `include/`, and `lib/` folders into your CUDA installation.
4. Add the CUDA `bin/` and `libnvvp/` paths to your Windows System Environment `PATH` variable.
5. Download `zlibwapi.lib` if missing, and place it in your CUDA `lib/` directory.

---

## Preprocess data & train the model

### 1. Preprocess the data
```bash
jupyter notebook Notebooks/LoadComicData_rgb.ipynb
```

### 2. Train the U-Net Model (RGB Version)
```bash
jupyter notebook Notebooks/train_unet_rgb_v2.ipynb
```

You can swap to LAB or HSV versions by running other notebooks, but are correctly broken / partcially unfucntional.

---
