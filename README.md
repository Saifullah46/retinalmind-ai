# 🧠 RetinalMind AI: Cognitive Risk Detection System

🚀 An AI-powered web application that analyzes retinal images to predict cognitive (dementia) risk using deep learning.

---

## 📌 Project Overview

RetinalMind AI is a deep learning-based system that analyzes retinal (fundus) images to detect potential cognitive decline. The model leverages transfer learning using ResNet18 to classify images into:

- ✅ No Significant Cognitive Risk  
- ⚠️ High Cognitive Risk Detected  

The system also provides a **confidence score** to improve interpretability.

---

## 🎯 Features

- 📤 Upload retinal images through web interface  
- 🧠 AI-based prediction using PyTorch (ResNet18)  
- 📊 Confidence score with progress visualization  
- 🎨 Modern UI with interactive elements  
- ⚡ Real-time prediction system  
- 📸 Image preview functionality  

---

## 🏗️ Tech Stack

- **Backend:** Python, Flask  
- **Machine Learning:** PyTorch, Torchvision  
- **Frontend:** HTML, CSS  
- **Image Processing:** PIL (Python Imaging Library)  

---

## 🧠 Model Details

- **Model Used:** ResNet18 (Pretrained)  
- **Task:** Binary Classification  
- **Classes:**
  - 0 → No Risk  
  - 1 → High Risk  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  

---

## 📂 Project Structure
retinalmind-ai/
│
├── app.py # Flask application
├── train_model.py # Model training script
├── predict.py # Prediction script
├── retinal_model.pth # Trained model
├── requirements.txt # Dependencies
│
├── templates/
│ └── index.html # Frontend UI
│
├── static/ # Uploaded images
│
└── README.md

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository

git clone https://github.com/YOUR_USERNAME/retinalmind-ai.git
cd retinalmind-ai
###2️⃣ Install dependencies
pip install -r requirements.txt
###3️⃣ Run the application
python app.py
###4️⃣ Open in browser
http://127.0.0.1:5000
###📊 Sample Output:
🚨 High Risk Detected
📈 Confidence: 87.45%

###Screenshot 
<img width="1366" height="617" alt="Screenshot 2026-04-28 214619" src="https://github.com/user-attachments/assets/a9fd6c3d-c353-4000-a5d0-2626b4368803" />

<img width="1366" height="618" alt="Screenshot 2026-04-28 214710" src="https://github.com/user-attachments/assets/6155dd01-28c3-48a9-b527-b16a3c38ce86" />


###⚠️ Dataset Disclaimer

The dataset used (ODIR - Ocular Disease Intelligent Recognition) is not included in this repository due to licensing restrictions.

###🚀 Future Enhancements
📱 Mobile application support
📊 Multi-class disease detection
☁️ Cloud deployment
🧠 Improved model accuracy with larger datasets


###👨‍💻 Author

Saifullah
B.Tech AI & Data Science
