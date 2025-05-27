# 🩺 Diabetes Prediction using MLP and TensorFlow

This project uses a Multi-Layer Perceptron (MLP) neural network built with TensorFlow/Keras to predict the likelihood of diabetes based on patient diagnostic data. It also includes a Streamlit web app for interactive predictions.

---

## 📊 Dataset

- **Source:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Description:** Medical diagnostic measurements of Pima Indian women, including features such as glucose level, BMI, age, etc.
- **Target:** Binary variable indicating diabetes outcome (1 = diabetic, 0 = non-diabetic)

---

## 🔧 Features

- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Pregnancies

---

## 🧠 Model Architecture

- Input Layer: 8 features
- Hidden Layers: Dense layers with ReLU activation + Dropout 
- Output Layer: 1 neuron with Sigmoid activation
- Optimizer: Adam
- Loss: Binary Crossentropy

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- Confusion Matrix
- Learning Curves

---


