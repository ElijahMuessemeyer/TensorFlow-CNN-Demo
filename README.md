# TensorFlow CNN Demo

A hands-on demonstration of Convolutional Neural Networks (CNNs) for handwritten digit recognition using TensorFlow and the MNIST dataset.

## Overview

This project implements a CNN that achieves ~99% accuracy on the MNIST handwritten digit classification task. It demonstrates deep learning fundamentals including convolutional layers, pooling, and fully connected networks.

## Project Structure

```
├── 01_verify_setup.ipynb          # Environment verification notebook
├── convolutional_network.ipynb    # Main CNN implementation
├── convolutional_network_executed.ipynb  # Notebook with outputs
├── executed_cnn.ipynb             # Additional executed notebook
├── TensorFlow_CNN_Essay.md        # Detailed analysis and documentation
├── QUICKSTART.txt                 # Quick setup guide
├── start_jupyter.sh               # Launch script (Unix)
└── start_jupyter.command          # Launch script (macOS)
```

## Model Architecture

- **Input Layer**: 28x28 grayscale images
- **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Flatten + Dense**: 128 units, ReLU activation
- **Output Layer**: 10 units (digits 0-9), Softmax activation

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install tensorflow jupyter numpy matplotlib

# Launch Jupyter
jupyter notebook convolutional_network.ipynb
```

Or use the provided launch scripts:
```bash
./start_jupyter.sh
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Jupyter Notebook
- NumPy
- Matplotlib

## Results

The trained model achieves approximately **99% accuracy** on the MNIST test set after training, demonstrating the effectiveness of CNNs for image classification.

## Key Concepts Demonstrated

- **Convolutional Layers**: Automatic feature extraction from images
- **Pooling Layers**: Spatial dimensionality reduction
- **Dropout**: Regularization to prevent overfitting
- **Adam Optimizer**: Adaptive learning rate optimization
- **Cross-Entropy Loss**: Standard loss function for classification

## Further Reading

See `TensorFlow_CNN_Essay.md` for a detailed analysis including:
- Installation process on Apple Silicon
- Model architecture explanation
- Applications of CNNs in computer vision
- Potential improvements and extensions
- Comparison with other datasets (Fashion-MNIST, CIFAR-10)

## License

MIT License
