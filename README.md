# 🌿 Plant Disease Recognition System
This project is a web-based system that uses machine learning to detect plant diseases from leaf images.
## Models Used
CNN (Convolutional Neural Network)** for image classification
Random Forest** on CNN-extracted features for improved reliability

## Features
- Upload or capture leaf images
- Top-3 prediction confidence
- Compare CNN vs. RF predictions
- View prediction history with timestamps
- Download results as CSV
- Multilingual interface (English & Amharic)

## echnologies
- Python, TensorFlow, scikit-learn
- Streamlit for web interface

## 📁 Folder Structure
```
├── main.py                 # Streamlit app
├── trained_plant_disease_model.keras
├── rf_model.pkl, scaler.pkl
├── test/, train/, valid/  # Dataset directories
├── output.png, confusion matrix.png
├── Train_plant_disease.ipynb
├── README.md, .gitignore, requirements.txt
```

## 🏁 How to Run
```bash
pip install -r requirements.txt
streamlit run main.py
```

