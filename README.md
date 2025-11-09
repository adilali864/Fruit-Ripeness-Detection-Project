# 🍎🍌 Fruit Ripeness Detection System

A real-time computer vision application that combines deep learning and object detection to identify fruits and classify their ripeness stages. This project leverages YOLOv3 for fruit detection and a custom CNN model for ripeness classification, providing instant feedback through a web-based interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-black.svg)
![License](https://img.shields.io/badge/License-Personal-red.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Results & Insights](#results--insights)
- [Future Enhancements](#future-enhancements)
- [About This Project](#about-this-project)

## 🎯 Overview

This project addresses a practical challenge in food quality assessment by automating the detection and classification of fruit ripeness. The system processes real-time video streams to identify fruits (currently supporting bananas and apples) and classifies them into three ripeness categories:
- **Unripe** - Fruit is not yet ready for consumption
- **Ripe** - Optimal condition for consumption
- **Overripe** - Past optimal freshness

The application has potential use cases in retail quality control, supply chain management, and consumer applications.

## ✨ Key Features

### Real-Time Detection
- Live video feed processing with minimal latency
- Simultaneous detection and classification of multiple fruits
- Instant ripeness assessment with confidence scores

### Dual-Model Architecture
- **YOLOv3 Object Detection**: Identifies and localizes fruits in the frame
- **Custom CNN Classifier**: Analyzes fruit regions to determine ripeness stage

### Web-Based Interface
- Flask-powered web application for easy access
- Clean, intuitive interface for viewing detection results
- Real-time video streaming with overlay annotations

### Advanced Image Processing
- Comprehensive data augmentation pipeline
- Normalized image preprocessing
- Non-Maximum Suppression (NMS) for accurate bounding boxes

## 🏗️ Technical Architecture

### Detection Pipeline
```
Video Frame → YOLOv3 Detection → Fruit Localization → ROI Extraction → 
CNN Classification → Ripeness Prediction → Annotated Output
```

### Model Architecture

**Object Detection (YOLOv3)**
- Pre-trained on COCO dataset
- Fine-tuned for fruit detection
- 416x416 input resolution
- Confidence threshold: 0.5

**Ripeness Classification (Custom CNN)**
```
Input (128x128x3)
    ↓
Conv Block 1 (32 filters) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv Block 2 (64 filters) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv Block 3 (128 filters) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Output (3 classes - Softmax)
```

## 💻 Technologies Used

### Core Technologies
- **Python 3.8+** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework for model development
- **OpenCV** - Computer vision and image processing
- **NumPy** - Numerical computing and array operations

### Web Framework
- **Flask** - Web application framework
- **HTML/CSS** - Frontend interface

### Model Components
- **YOLOv3** - Object detection (Darknet framework)
- **Custom CNN** - Ripeness classification
- **ImageDataGenerator** - Data augmentation

### Development Tools
- **ModelCheckpoint** - Model versioning
- **EarlyStopping** - Training optimization
- **ReduceLROnPlateau** - Learning rate scheduling
- **TensorBoard Compatible** - Training visualization

## 📁 Project Structure

```
Fruit-Ripeness-Detection-Project/
│
├── detect.py                    # Main Flask application & detection logic
├── train_model.py              # Model training script with advanced callbacks
├── prepare_dataset.py          # Dataset preparation and preprocessing
│
├── templates/
│   └── index.html              # Web interface template
│
├── model_outputs/
│   ├── final_model.keras       # Trained ripeness classification model
│   ├── best_model.keras        # Best model checkpoint during training
│   ├── yolov3.weights          # YOLOv3 pre-trained weights
│   ├── yolov3.cfg              # YOLOv3 configuration
│   └── coco.names              # COCO dataset class names
│
├── processed_dataset/
│   ├── X_train.npy             # Training images
│   ├── X_test.npy              # Testing images
│   ├── y_fruit_train.npy       # Training labels
│   └── y_fruit_test.npy        # Testing labels
│
├── logs/                       # Training logs and metrics
│
└── README.md                   # Project documentation
```

## 📊 Model Performance

### Training Configuration
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 32
- **Optimizer**: Adam
- **Initial Learning Rate**: 0.001
- **Learning Rate Decay**: Step decay (0.5 every 10 epochs)
- **Minimum Learning Rate**: 1e-5

### Data Augmentation Strategy
- Rotation: ±20°
- Width/Height Shift: 20%
- Horizontal & Vertical Flips
- Zoom Range: 20%
- Shear Range: 20%
- Brightness Variation: 80-120%

### Evaluation Metrics
The model is evaluated using:
- **Accuracy** - Overall classification correctness
- **Precision** - Measure of positive prediction accuracy
- **Recall** - Measure of actual positive detection
- **Loss** - Categorical cross-entropy

### Callbacks & Optimization
- **EarlyStopping**: Monitors validation loss with patience of 10 epochs
- **ReduceLROnPlateau**: Reduces learning rate by 50% after 5 epochs without improvement
- **ModelCheckpoint**: Saves best model based on validation accuracy

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
Webcam/Camera device
Minimum 8GB RAM recommended
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/adilali864/Fruit-Ripeness-Detection-Project.git
cd Fruit-Ripeness-Detection-Project
```

### Step 2: Install Dependencies
```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install flask
```

### Step 3: Download YOLOv3 Files
Download the following files and place them in the `model_outputs/` directory:

1. **yolov3.weights** (237MB)
   - Download from: https://pjreddie.com/media/files/yolov3.weights

2. **yolov3.cfg**
   - Download from: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

3. **coco.names**
   - Download from: https://github.com/pjreddie/darknet/blob/master/data/coco.names

### Step 4: Prepare Dataset (If Training)
```bash
python prepare_dataset.py
```

### Step 5: Train Model (Optional)
```bash
python train_model.py
```
Training will generate:
- `final_model.keras` - Final trained model
- `best_model.keras` - Best checkpoint during training
- Training logs in the `logs/` directory

## 🎮 Usage

### Running the Application

1. **Start the Flask Server**
```bash
python detect.py
```

2. **Access the Web Interface**
Open your browser and navigate to:
```
http://localhost:5000
```

3. **Using the Application**
- The webcam feed will automatically start
- Point your camera at fruits (bananas or apples)
- The system will draw bounding boxes around detected fruits
- Ripeness classification and confidence scores will be displayed in real-time

### Application Controls
- The video feed updates in real-time
- Multiple fruits can be detected simultaneously
- Confidence scores are displayed as percentages
- Color-coded bounding boxes indicate detection

## 🔍 How It Works

### Detection Process

1. **Frame Capture**
   - Captures frames from the webcam at 30 FPS
   - Preprocesses frames for YOLO input (416x416)

2. **Object Detection**
   - YOLOv3 processes the frame
   - Identifies potential fruit objects (banana, apple)
   - Filters detections with confidence > 0.5
   - Applies NMS to eliminate duplicate detections

3. **Region of Interest (ROI) Extraction**
   - Extracts fruit regions from bounding boxes
   - Ensures coordinates are within frame boundaries
   - Handles edge cases for partial detections

4. **Ripeness Classification**
   - Resizes ROI to 128x128 pixels
   - Normalizes pixel values (0-1 range)
   - Feeds into CNN classifier
   - Outputs probabilities for three classes

5. **Result Visualization**
   - Draws bounding boxes on original frame
   - Adds labels with fruit type, ripeness stage, and confidence
   - Streams annotated frames to web interface

### Training Process

1. **Data Preparation**
   - Dataset loading and validation
   - Train-test split
   - Image normalization

2. **Model Architecture Setup**
   - Three convolutional blocks with increasing depth
   - Batch normalization for training stability
   - Dropout layers for regularization
   - Dense layers for classification

3. **Training Optimization**
   - Data augmentation on-the-fly
   - Learning rate scheduling
   - Early stopping to prevent overfitting
   - Model checkpointing for best weights

4. **Evaluation & Saving**
   - Validation on test set
   - Metrics calculation (accuracy, precision, recall)
   - Model serialization in Keras format

## 📈 Results & Insights

### Key Achievements

✅ Successfully integrated two complementary models for end-to-end fruit analysis  
✅ Achieved real-time performance suitable for practical applications  
✅ Implemented robust error handling for production-ready deployment  
✅ Created an intuitive web interface for easy accessibility  
✅ Developed a scalable architecture for future fruit types  

### Technical Insights

**Model Convergence**
- Early stopping typically triggers around epoch 30-40
- Learning rate reduction helps fine-tune in later epochs
- Batch normalization significantly improves training stability

**Detection Accuracy**
- YOLOv3 provides reliable fruit localization with minimal false positives
- NMS threshold of 0.4 balances precision and recall effectively
- Confidence threshold of 0.5 filters unreliable detections

**Ripeness Classification**
- Color features play a crucial role in ripeness determination
- Data augmentation prevents overfitting on limited datasets
- Dropout layers improve generalization to new fruit samples

### Challenges Overcome

1. **Real-time Processing**: Optimized model inference and frame processing for smooth video streams
2. **Varying Lighting**: Data augmentation with brightness variations improved robustness
3. **Multiple Fruits**: NMS algorithm handles overlapping detections effectively
4. **Edge Cases**: Comprehensive error handling for boundary conditions

## 🔮 Future Enhancements

### Planned Improvements

- [ ] **Expand Fruit Support**: Add support for oranges, mangoes, tomatoes, and other common fruits
- [ ] **Mobile Application**: Develop iOS/Android apps for on-the-go fruit assessment
- [ ] **Model Optimization**: Implement model quantization for faster inference
- [ ] **Dataset Expansion**: Collect more diverse training data for improved accuracy
- [ ] **Advanced Metrics**: Add shelf-life prediction and nutritional value estimation
- [ ] **Cloud Integration**: Deploy model on cloud platforms for scalable access
- [ ] **Batch Processing**: Add support for analyzing multiple images/videos
- [ ] **Multi-language Support**: Internationalize the interface for global use

### Potential Features

- **Export Functionality**: Save detection results and generate reports
- **Analytics Dashboard**: Track ripeness trends and statistics over time
- **API Endpoints**: RESTful API for integration with other applications
- **Notification System**: Alerts for fruits reaching optimal ripeness
- **Database Integration**: Store historical data for analysis

## 📝 About This Project

### Personal Learning Journey

This project represents a comprehensive exploration of computer vision and deep learning technologies. Through its development, I gained hands-on experience with:

- **Deep Learning**: Designing, training, and optimizing CNN architectures
- **Computer Vision**: Implementing object detection and image classification pipelines
- **Web Development**: Creating full-stack applications with Flask
- **Model Deployment**: Integrating ML models into production applications
- **Software Engineering**: Writing clean, maintainable, and well-documented code

### Project Context

Developed as a personal portfolio project to demonstrate proficiency in:
- Machine Learning & Deep Learning
- Computer Vision Applications
- Full-Stack Development
- Project Management & Documentation

This project showcases the ability to take an idea from concept to fully functional application, incorporating industry best practices and modern development workflows.

### Development Timeline

- **Phase 1**: Research and architecture design
- **Phase 2**: Dataset collection and preprocessing
- **Phase 3**: Model development and training
- **Phase 4**: Web application development
- **Phase 5**: Testing, optimization, and documentation

---

## 📫 Contact & Professional Profile

**Adil Khan**
- GitHub: [@adilali864](https://github.com/adilali864)
- Twitter: [@Adilali864](https://twitter.com/Adilali864)

---

## 📄 Project Status

**Status**: ✅ Complete - Fully Functional  
**Version**: 1.0.0  
**Last Updated**: November 2024

---

## ⚖️ License & Usage

This is a personal portfolio project created for educational and demonstration purposes. The code and documentation are provided as-is for reference.

**Note**: This project is not open for external contributions as it serves as a personal portfolio piece demonstrating individual technical capabilities.

---

## 🙏 Acknowledgments

- YOLOv3 by Joseph Redmon and Ali Farhadi
- TensorFlow and Keras development teams
- OpenCV community
- COCO dataset contributors

---

**Built with ❤️ by Adil Khan**