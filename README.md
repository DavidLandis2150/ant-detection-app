# 🐜 Random Forest Ant Detection System

A Streamlit-based application for detecting ants in images using Random 
Forest machine learning.

## Features
- Image annotation tool for creating training data
- Random Forest model training with customizable parameters
- Real-time ant detection in uploaded images
- Model saving and loading capabilities

## Live Demo
[Try the app here](https://your-app-name.streamlit.app)

## How to Use
1. **Annotation Mode**: Upload images and draw bounding boxes around ants
2. **Training Mode**: Upload annotated images to train the Random Forest 
model
3. **Prediction Mode**: Load a trained model and detect ants in new images

## Local Installation
```bash
pip install -r requirements.txt
streamlit run app.py
