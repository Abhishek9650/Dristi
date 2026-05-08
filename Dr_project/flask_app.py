from flask import Flask, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import tensorflow as tf
import numpy as np
import cv2
import os

from datetime import datetime

from tensorflow.keras.applications.efficientnet import preprocess_input

from utils.db import users, history, patient_data
from utils.report_generator import create_report
import joblib

# =========================
# LOAD ENV
# =========================

load_dotenv()

# =========================
# FLASK APP
# =========================

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")

# =========================
# FOLDERS
# =========================

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "best_model.keras"
)

# Diabetes prediction model

diabetes_model = joblib.load(
    "model/diabetes_model.pkl"
)

scaler = joblib.load(
    "model/scaler2.pkl"
)

# =========================
# CLASSES
# =========================

classes = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

# =========================
# ALLOWED FILES
# =========================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

def allowed_file(filename):

    return "." in filename and \
    filename.rsplit(".",1)[1].lower() \
    in ALLOWED_EXTENSIONS

# =========================
# PREDICTION FUNCTION
# =========================

def predict_image(path):

    img = cv2.imread(path)

    # Convert grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Remove black borders
    _, thresh = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    x, y, w, h = cv2.boundingRect(thresh)

    img = img[y:y+h, x:x+w]

    # Resize
    img_resized = cv2.resize(
        img,
        (300,300)
    )

    # Preprocess
    img_array = np.expand_dims(
        img_resized,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    # Prediction
    preds = model.predict(img_array)[0]

    pred_class = np.argmax(preds)

    confidence = round(
        float(preds[pred_class]) * 100,
        2
    )

    prediction = classes[pred_class]

    # All probabilities
    prediction_data = {

        classes[i]:
        round(float(preds[i])*100,2)

        for i in range(len(classes))
    }

    return (
        prediction,
        confidence,
        prediction_data
    )

# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template("index.html")

# =========================
# SIGNUP
# =========================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        # =========================
        # VALIDATION
        # =========================

        if not name or not email or not password:
            return render_template(
                "signup.html",
                error="All fields are required"
            )

        if len(password) < 6:
            return render_template(
                "signup.html",
                error="Password must be at least 6 characters"
            )

        # =========================
        # CHECK EXISTING USER
        # =========================

        existing = users.find_one({
            "email": email
        })

        if existing:
            return render_template(
                "signup.html",
                error="User already exists"
            )

        # =========================
        # HASH PASSWORD
        # =========================

        hashed = generate_password_hash(password)

        # =========================
        # INSERT USER
        # =========================

        users.insert_one({

            "name": name,
            "email": email,
            "password": hashed
        })

        return redirect("/login")

    return render_template("signup.html")

# =========================
# LOGIN
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        # Find user in DB
        user = users.find_one({
            "email": email
        })

        # Check user + password
        if user and check_password_hash(user["password"], password):

            session["user"] = user["email"]
            session["name"] = user["name"]

            return redirect("/dashboard")

        # If invalid
        return render_template(
            "login.html",
            error="Invalid email or password"
        )

    return render_template("login.html")
# =========================
# DASHBOARD
# =========================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:

        return redirect("/login")

    return render_template(
        "dashboard.html",
        name=session["name"]
    )

# =========================
# PREDICT
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect("/login")

    # =========================
    # DIABETES PREDICTION
    # =========================

    pregnancies = float(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    bloodpressure = float(request.form["bloodpressure"])
    skinthickness = float(request.form["skinthickness"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    dpf = float(request.form["dpf"])
    age = float(request.form["age"])

    diabetes_features = np.array([[
        pregnancies,
        glucose,
        bloodpressure,
        skinthickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale input
    diabetes_features = scaler.transform(diabetes_features)

    # Prediction
    diabetes_pred = diabetes_model.predict(diabetes_features)[0]
    diabetes_prob = diabetes_model.predict_proba(diabetes_features)[0][1]

    diabetes_percent = round(float(diabetes_prob) * 100, 2)
    diabetes_result = "Diabetic" if diabetes_pred == 1 else "Non-Diabetic"

    # =========================
    # SAVE PATIENT DATA (NEW)
    # =========================

    patient_data.insert_one({

        "user": session["user"],

        "pregnancies": pregnancies,
        "glucose": glucose,
        "bloodpressure": bloodpressure,
        "skinthickness": skinthickness,
        "insulin": insulin,
        "bmi": bmi,
        "dpf": dpf,
        "age": age,

        "diabetes_result": diabetes_result,
        "diabetes_percent": diabetes_percent,

        "time": datetime.now()
    })

    # =========================
    # IMAGE VALIDATION
    # =========================

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    if not allowed_file(file.filename):
        return "Invalid file type"

    filename = secure_filename(file.filename)

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # =========================
    # DR PREDICTION
    # =========================

    prediction, confidence, prediction_data = predict_image(filepath)

    # =========================
    # HYBRID LOGIC (BONUS)
    # =========================

    risk_score = (diabetes_percent * 0.6) + (confidence * 0.4)

    if risk_score > 75:
        final_risk = "High Risk"
    elif risk_score > 40:
        final_risk = "Moderate Risk"
    else:
        final_risk = "Low Risk"

    if diabetes_result == "Diabetic" and prediction == "No DR":
        final_stage = "Early Risk (No DR visible)"
    else:
        final_stage = prediction

    # =========================
    # REPORT
    # =========================

    report_name = filename.split(".")[0] + "_report.pdf"
    report_path = os.path.join(REPORT_FOLDER, report_name)

    create_report(
        report_path,
        session["name"],
        age,   # 👈 IMPORTANT (get from form)
        prediction,
        diabetes_result
    )

    # =========================
    # SAVE HISTORY
    # =========================

    history.insert_one({

        "user": session["user"],

        "diabetes_result": diabetes_result,
        "diabetes_percent": diabetes_percent,

        "prediction": final_stage,
        "confidence": confidence,
        "risk": final_risk,

        "prediction_data": prediction_data,

        "report": report_path,
        "image": filepath,

        "time": datetime.now()
    })

    # =========================
    # RESULT PAGE
    # =========================

    return render_template(

        "result.html",

        diabetes_result=diabetes_result,
        diabetes_percent=diabetes_percent,

        prediction=final_stage,
        confidence=confidence,

        final_risk=final_risk,

        image=filepath,
        report=report_path
    )
# =========================
# HISTORY
# =========================

@app.route("/history")
def history_page():

    if "user" not in session:

        return redirect("/login")

    records = history.find({

        "user": session["user"]

    }).sort("time",-1)

    return render_template(

        "history.html",

        records=records
    )

# =========================
# LOGOUT
# =========================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")

# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(debug=True)