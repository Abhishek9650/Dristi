# 🩺 Diabetic Retinopathy Detection System

An AI-powered web application for detecting **Diabetic Retinopathy (DR)** from retinal fundus images using Deep Learning and Machine Learning techniques.

---

# 📌 Overview

Diabetic Retinopathy is a diabetes-related eye disease that can lead to blindness if not detected early. This project uses a hybrid AI approach combining:

* **EfficientNetB3** for deep feature extraction
* **XGBoost** for final classification
* **Flask** for web deployment
* **Grad-CAM** for explainable AI visualization

The system predicts the severity stage of diabetic retinopathy from uploaded retinal images.

---

# 🚀 Features

✅ Upload retinal fundus images
✅ AI-based DR stage prediction
✅ Confidence score display
✅ Grad-CAM heatmap visualization
✅ Modern responsive UI
✅ Fast prediction system
✅ Web-based deployment using Flask

---

# 🧠 Technologies Used

| Technology          | Purpose                   |
| ------------------- | ------------------------- |
| Python              | Core Programming Language |
| TensorFlow / Keras  | Deep Learning             |
| EfficientNetB3      | Feature Extraction        |
| XGBoost             | Classification            |
| Flask               | Backend Web Framework     |
| HTML/CSS/JavaScript | Frontend Development      |
| OpenCV              | Image Processing          |
| NumPy               | Numerical Operations      |
| Grad-CAM            | Model Explainability      |

---

# 📂 Project Structure

```bash
DR_Project/
│
├── model/
│   ├── dr_model_final2.keras
│   └── xgboost_model.pkl
│
├── static/
│   ├── uploads/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

# 🖼️ DR Classification Stages

The model classifies retinal images into the following categories:

| Class | Description      |
| ----- | ---------------- |
| 0     | No DR            |
| 1     | Mild DR          |
| 2     | Moderate DR      |
| 3     | Severe DR        |
| 4     | Proliferative DR |

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/dr-project.git
cd dr-project
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python app.py
```

Open browser:

```bash
http://127.0.0.1:5000/
```

---

# 🧪 Model Training

To train the model:

```bash
python train.py
```

Training includes:

* Data preprocessing
* Image augmentation
* EfficientNetB3 feature extraction
* XGBoost classification
* Model evaluation

---

# 📊 Why EfficientNetB3?

EfficientNetB3 was selected because:

* Better accuracy with fewer parameters
* Efficient scaling of depth, width, and resolution
* Faster training compared to larger models
* Strong performance on medical imaging datasets
* Reduced overfitting risk

---

# 📈 Why XGBoost?

XGBoost was used because:

* High classification performance
* Handles extracted features effectively
* Reduces overfitting using regularization
* Faster and efficient boosting algorithm
* Improves final prediction accuracy

---

# 🔥 Grad-CAM Visualization

Grad-CAM is used to highlight important retinal regions influencing model predictions.

Benefits:

* Improves explainability
* Helps doctors trust AI predictions
* Visual interpretation of disease regions

---

# 📷 Dataset

Dataset used:

* APTOS 2019 Blindness Detection Dataset
* Kaggle Diabetic Retinopathy Dataset

Dataset contains retinal fundus images labeled according to DR severity.

---

# 📉 Evaluation Metrics

The model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

# 💡 Future Improvements

* User authentication system
* Cloud deployment
* Real-time camera scanning
* Multi-disease eye detection
* Improved Grad-CAM visualizations
* Mobile application support

---

# 🌐 Deployment

The project can be deployed using:

* Render
* Railway
* Heroku
* AWS
* Google Cloud

---

# 🛡️ Disclaimer

This project is developed for educational and research purposes only and should not replace professional medical diagnosis.

---

# 👨‍💻 Author

**Abhishek Kapoor and Pema Tsomo**  

AI & Machine Learning Developer

---

# ⭐ Acknowledgements

Special thanks to:

* TensorFlow Team
* Kaggle Datasets Community
* OpenCV Contributors
* Flask Community

---

# 📜 License

This project is licensed under the MIT License.
