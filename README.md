# Dermal Scan - AI Skin Aging Detection

## 🌐Live Demo: https://abhinav-dermal-scan.streamlit.app/

An AI-powered application for analyzing facial skin aging signs using EfficientNetB0 deep learning model.

## 🏗️ Repository Structure

```
Dermal-Scan/
├── app.py                          # Main Streamlit application
├── components.py                   # UI components and rendering logic
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages
├── README.md                       # This file
├── test_enhanced.py               # Enhanced testing script
│
├── assets/                        # Static assets
│   ├── css/
│   │   └── styles.css             # Custom CSS styling
│   └── images/                    # Static image assets
│
├── src/                           # Source code directory
│   ├── __init__.py
│   ├── train_dermal_scan.py       # Comprehensive training pipeline
│   │
│   ├── models/                    # Model-related modules
│   │   ├── __init__.py
│   │   ├── model_building.py      # EfficientNetB0 model architecture
│   │   └── training/
│   │       ├── train.py           # Training utilities and functions
│   │       └── training_accuracy.py
│   │
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # EfficientNetB0-compatible preprocessing
│   │   └── augmentation.py        # Data augmentation utilities
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       └── class_weights.py       # Class weight calculation utilities
│
├── models/                        # Trained model files
│   ├── dermal_scan_last.h5        # Feature classification model
│   ├── age_pred.h5                # Age prediction model
│   ├── predict_age.py             # Age prediction utilities
│   ├── predict_feature.py         # Feature prediction utilities
│   ├── test.py                    # Model testing script
│   ├── test2.py                   # Additional testing script
│   └── old_models/                # Previous model versions
│       ├── age_net.caffemodel     # Legacy Caffe model
│       ├── dherma_ai_scan_v1.h5   # Previous H5 model version
│       └── dherma_ai_scan_v1.keras # Previous Keras model version
│
├── image_load/                    # Image loading and preprocessing
│   ├── loader.py                  # Image loading utilities
│   ├── preprocess.py              # Real-time image preprocessing
│   ├── label.py                   # Labeling and annotation
│   └── haarcascade_frontalface_default.xml # OpenCV face detection cascade
│
├── sample_images/                 # Sample test images
│   ├── acne.jpg
│   ├── girl1.jpg
│   ├── girl2.jpg
│   ├── kid1.jpg
│   ├── kid2.jpg
│   ├── man1.jpg
│   ├── man2.jpg
│   ├── puffy_eye_man.jpg
│   ├── srk.jpg
│   └── woman.png
│
└── logs/                          # Application logs
    └── dermal_scan_*.log          # Timestamped log files
```

## 🚀 Features

- **AI-Powered Analysis**: Uses EfficientNetB0 for accurate skin condition classification
- **Multiple Conditions**: Detects wrinkles, dark spots, puffy eyes, and clear face
- **Age Prediction**: Estimates facial age using deep learning
- **Real-time Processing**: Fast inference with optimized preprocessing
- **Interactive UI**: User-friendly Streamlit interface
- **Data Augmentation**: Built-in tools for dataset balancing

## 🔧 Model Architecture

### EfficientNetB0 Features Model
- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Input Size**: 224x224x3
- **Output Classes**: 4 (clear_face, darkspots, puffy_eyes, wrinkles)
- **Training Strategy**: Two-phase transfer learning
- **Preprocessing**: EfficientNet-specific preprocessing pipeline

### Age Prediction Model
- **Architecture**: Custom CNN for regression
- **Input Size**: 224x224x3
- **Output**: Single continuous value (age)
- **Loss Function**: Mean Squared Error

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Dermal-Scan
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🎯 Training Your Own Model

### Quick Start Training

```bash
cd Dermal-Scan
python src/train_dermal_scan.py --dataset_path /path/to/your/dataset --num_classes 4
```

### Advanced Training Options

```bash
python src/train_dermal_scan.py \
    --dataset_path /path/to/dataset \
    --num_classes 4 \
    --batch_size 32 \
    --epochs 30 \
    --fine_tune_epochs 20 \
    --learning_rate 0.001 \
    --output_dir outputs \
    --model_name dermal_scan_efficientnet
```

### Dataset Structure

Your dataset should be organized as follows:

```
dataset/
├── clear_face/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── darkspots/
│   ├── image1.jpg
│   └── ...
├── puffy_eyes/
│   └── ...
└── wrinkles/
    └── ...
```

### Data Augmentation

Balance your dataset using the augmentation utilities:

```python
from src.data.augmentation import augment_minority_classes, print_class_distribution

# Balance dataset
final_distribution = augment_minority_classes("/path/to/dataset")
print_class_distribution(final_distribution)
```

## 📊 Training Pipeline Features

The comprehensive training pipeline includes:

- **Two-Phase Training**: 
  1. Phase 1: Frozen EfficientNetB0 base
  2. Phase 2: Fine-tuning with unfrozen base
- **Automatic Class Balancing**: Handles imbalanced datasets
- **Advanced Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **TensorBoard Integration**: Real-time training monitoring
- **Comprehensive Logging**: Detailed training history and metrics
- **Visualization**: Automatic generation of training plots
- **Model Evaluation**: Built-in validation and testing

## 🔍 Model Performance

### Training Metrics
- **Architecture**: EfficientNetB0 Transfer Learning
- **Training Strategy**: Two-Phase (Frozen → Fine-tune)
- **Input Resolution**: 224×224
- **Preprocessing**: EfficientNet-optimized pipeline
- **Data Augmentation**: Rotation, shifts, zoom, brightness, horizontal flip

### Expected Performance
- **Training Accuracy**: 85-95% (depending on dataset quality)
- **Validation Accuracy**: 80-90%
- **Inference Time**: <100ms per image
- **Model Size**: ~20MB (EfficientNetB0)

## 📝 Usage Examples

### Using Individual Components

```python
# Model building
from src.models.model_building import build_efficientnet_model, compile_model

model = build_efficientnet_model(num_classes=4)
model = compile_model(model, learning_rate=1e-4)

# Data preprocessing
from src.data.preprocessing import create_data_generators

train_gen, val_gen = create_data_generators('/path/to/dataset')

# Training
from src.models.training.train import train_efficientnet_model

model, history = train_efficientnet_model(
    dataset_path='/path/to/dataset',
    num_classes=4,
    model_save_path='my_model.h5'
)
```

### Custom Training Configuration

```python
# Advanced training with custom parameters
model, history = train_efficientnet_model(
    dataset_path='/path/to/dataset',
    num_classes=4,
    model_save_path='custom_model.h5',
    epochs=40,
    batch_size=16,
    fine_tune_epochs=25
)
```

## 🧪 Testing and Evaluation

Run tests on sample images:

```bash
# Test with sample images
python models/test.py

# Test age prediction
python models/test2.py
```

## 📦 Model Files

- `models/dermal_scan_last.h5`: Main feature classification model
- `models/age_pred.h5`: Age prediction model
- `models/old_models/`: Previous versions and experimental models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- EfficientNetB0 architecture by Google Research
- Streamlit for the web interface
- OpenCV for image processing
- TensorFlow/Keras for deep learning

## 📞 Support

For questions and support, please open an issue in the GitHub repository.

---

**© 2025 DermalScan AI | Powered by EfficientNetB0 & Streamlit**