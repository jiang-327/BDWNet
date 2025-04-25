# Wood Defect Detection

This project uses deep learning methods for wood defect detection and segmentation.

## Project Structure

```
├── configs/               # Configuration files
│   └── config.py         # Parameter configuration
├── data/                 # Data directory
│   ├── train/           # Training set
│   ├── val/             # Validation set
│   └── test/            # Test set
├── models/              # Model definitions
│   ├── loss.py         # Loss function definitions
│   └── network.py      # Network structure definitions
├── utils/              # Utility functions
│   ├── dataset.py      # Dataset loading and preprocessing
│   └── metrics.py      # Evaluation metrics calculation
├── train.py           # Training script
└── test.py            # Testing script
```

## Usage

### 1. Environment Setup

Make sure to install the following dependencies:
```
pip install torch torchvision numpy opencv-python matplotlib albumentations scikit-learn tqdm
```

### 2. Data Preparation

Please organize your dataset according to the following structure:
```
data/
├── train/              # Training set images and masks
├── val/                # Validation set images and masks
└── test/               # Test set images and masks
```

Each directory should contain:
- images/: Image files
- masks/: Corresponding mask files

### 3. Training the Model

Run the following command to start training:
```
python train.py
```

### 4. Testing the Model

After training is complete, run the following command to test model performance:
```
python test.py
```

## Main File Descriptions

- `models/network.py`: WoodDefectBD network model definition
- `models/loss.py`: Loss function definitions
- `utils/dataset.py`: Data loading and preprocessing
- `utils/metrics.py`: Model evaluation metrics
- `configs/config.py`: Training and model parameter configuration
- `train.py`: Training script
- `test.py`: Testing script 