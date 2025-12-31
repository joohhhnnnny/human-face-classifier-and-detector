#  Human Face Classifier and Detector

A powerful deep learning application that detects and classifies human faces in images using **MobileNetV2** and **YOLOv8** models. Built with Streamlit for an intuitive web interface.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

---

##  Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models](#-models)
- [Training](#-training)
- [API Reference](#-api-reference)
- [License](#-license)

---

##  Features

- **Dual Model Support**: Choose between MobileNetV2 (classification) or YOLOv8 (object detection)
- **Real-time Analysis**: Upload images and get instant predictions
- **Visual Feedback**: Bounding boxes and confidence scores displayed on detected humans
- **User-friendly Interface**: Clean, responsive Streamlit web application
- **High Accuracy**: Models trained on custom datasets for optimal performance

---

##  Demo

1. Upload an image (JPG, JPEG, or PNG)
2. Select your preferred AI model from the sidebar
3. Click "Analyze Image" to see the results
4. View detection results with confidence scores

---

##  Project Structure

```
human-face-classifier-and-detector/
├── apt.py                     # Main Streamlit app (uses saved .h5 model)
├── sub.py                     # Alternative app (uses ImageNet weights)
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python version specification
├── packages.txt              # System-level dependencies
├── LICENSE                   # Apache 2.0 License
├── models/
│   ├── mobilenetv2.h5        # Trained MobileNetV2 model
│   ├── mobilenetv2.keras     # Keras format model
│   ├── mobilenetv2_saved.keras
│   └── best-yolov8s-v2.pt    # Trained YOLOv8s model
└── ipynb/
    ├── MobileNetv2_Train.ipynb   # MobileNetV2 training notebook
    └── Yolov8s_Train.ipynb       # YOLOv8s training notebook
```

---

##  Installation

### Prerequisites

- Python 3.11
- pip or conda package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/human-face-classifier-and-detector.git
cd human-face-classifier-and-detector
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install System Dependencies (Linux)

```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
```

---

##  Usage

### Running the Application

```bash
# Using the main application (with trained model)
streamlit run apt.py

# Using the alternative application (with ImageNet weights)
streamlit run sub.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Interface

1. **Select Model**: Use the sidebar to choose between:
   - **MobileNetV2**: Binary classification (Human vs Non-Human)
   - **YOLOv8**: Object detection with bounding boxes

2. **Upload Image**: Click "Upload an image" and select a JPG, JPEG, or PNG file

3. **Analyze**: Click the "Analyze Image" button to process

4. **View Results**: 
   - MobileNetV2: Shows Human/Non-Human classification with confidence score
   - YOLOv8: Displays bounding boxes around detected humans with count

---

##  Models

### MobileNetV2

- **Architecture**: Transfer learning with MobileNetV2 base
- **Input Size**: 224 × 224 × 3
- **Output**: Binary classification (Human probability)
- **Confidence Threshold**: 70%

### YOLOv8s

- **Architecture**: YOLOv8 Small variant
- **Task**: Object detection (Person class)
- **Output**: Bounding boxes with confidence scores
- **Features**: Real-time detection with visual annotations

---

##  Training

Training notebooks are available in the `ipynb/` directory:

### MobileNetV2 Training

The [MobileNetv2_Train.ipynb](ipynb/MobileNetv2_Train.ipynb) notebook includes:

- Data augmentation (rotation, shifts, zoom, flips)
- Transfer learning from ImageNet weights
- Custom classification head with dropout
- Binary cross-entropy loss optimization

```python
# Key training parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
```

### YOLOv8 Training

The [Yolov8s_Train.ipynb](ipynb/Yolov8s_Train.ipynb) notebook includes:

- Dataset preparation and splitting (80/20 train/val)
- YOLOv8s model fine-tuning
- Performance evaluation and visualization

---

##  API Reference

### Core Functions

#### `mobilenet_predict(image, model)`
Performs binary classification using MobileNetV2.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | PIL.Image | Input image |
| `model` | tf.keras.Model | Loaded MobileNetV2 model |

**Returns**: `float` - Human probability (0.0 to 1.0)

#### `yolo_detect_and_draw(image, model)`
Performs object detection and draws bounding boxes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | PIL.Image | Input image |
| `model` | YOLO | Loaded YOLOv8 model |

**Returns**: `tuple(np.ndarray, int)` - Annotated image and person count

---

##  Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.0 | Web interface |
| tensorflow-cpu | 2.15.0 | MobileNetV2 model |
| torch | 2.1.0 | PyTorch backend |
| torchvision | 0.16.0 | Vision utilities |
| ultralytics | 8.0.196 | YOLOv8 implementation |
| opencv-python-headless | 4.8.1.78 | Image processing |
| pillow | 10.1.0 | Image handling |
| numpy | 1.24.3 | Numerical operations |

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/) for the MobileNetV2 architecture
- [Ultralytics](https://ultralytics.com/) for the YOLOv8 implementation
- [Streamlit](https://streamlit.io/) for the web framework

---

## Project Contributors

- John Benedict Bongcac
- John Mhel Dalumpines
- Julie Anne Pesaña

---

<p align="center">
  Made with ❤️ for Computer Vision
</p>
