# 🧠 Handwritten Digit Classifier (CNN from Scratch using NumPy)

## 📌 Problem Statement
The objective of this project is to design and implement a Convolutional Neural Network (CNN) completely from scratch using only NumPy (without using deep learning frameworks like TensorFlow or PyTorch) to classify handwritten digits from the MNIST dataset.  
The project also includes deployment of the trained model using a Streamlit web interface for real-time digit prediction.

---

## 📊 Dataset Description
- Dataset Used: MNIST Handwritten Digit Dataset
- Total Samples: 70,000 images  
  - Training Set: 60,000 images  
  - Test Set: 10,000 images  
- Image Size: 28 × 28 pixels (Grayscale)
- Number of Classes: 10 (Digits 0–9)
- Format: IDX binary files (train-images-idx3-ubyte, train-labels-idx1-ubyte, etc.)

The dataset was manually loaded and preprocessed using a custom NumPy-based loader.

---

## 🏗️ Model Architecture (CNN from Scratch)
The CNN model was implemented without any deep learning libraries.  
All layers and operations were built manually using NumPy.

### Architecture:
Input (1 × 28 × 28)  
→ Convolution Layer (8 Filters, 3×3 Kernel)  
→ ReLU Activation  
→ Max Pooling Layer (2×2)  
→ Flatten Layer  
→ Fully Connected (Dense Layer)  
→ Softmax Output Layer  

This architecture allows the model to extract spatial features and classify handwritten digits effectively.

---

## ⚙️ Implementation Details
### Technologies Used:
- Python
- NumPy (Core computations)
- Streamlit (Deployment UI)
- Pillow (Image Processing)
- Pickle (Model Serialization)

### Key Components Implemented from Scratch:
- Convolution Layer (Forward Pass)
- Max Pooling (Forward & Backward Pass)
- Flatten Layer
- Dense Layer with Backpropagation
- ReLU Activation Function
- Softmax + Cross Entropy Loss
- Custom MNIST Data Loader (IDX format)
- Gradient Descent Optimizer

No external deep learning frameworks were used.

---

## 🔄 Data Preprocessing
- Normalization of pixel values (0–255 → 0–1)
- Reshaping to CNN input format (1, 28, 28)
- Noise removal for canvas-drawn digits
- Grayscale conversion for uploaded images

---

## 📈 Training Details
- Epochs: 3
- Learning Rate: 0.001
- Training Samples Used: 5000 (subset for faster training)
- Optimization: Gradient Descent (manual implementation)
- Loss Function: Softmax Cross Entropy Loss

---

## 📊 Evaluation Metrics
The model performance was evaluated using:

- Training Accuracy: ~89–90%
- Test Accuracy: ~85–88%
- Loss Reduction Across Epochs
- Prediction Confidence Scores

The model demonstrates good generalization on unseen MNIST test data despite being implemented from scratch.

---

## 🌐 Deployment (Streamlit Web App)
A Streamlit-based user interface was developed to demonstrate real-time predictions.

### Features:
- Upload handwritten digit images (28×28)
- Draw digits on canvas
- Real-time prediction with confidence score
- Visualization of processed 28×28 input image

This shows end-to-end pipeline from training to deployment.

---

## 📁 Project Structure
