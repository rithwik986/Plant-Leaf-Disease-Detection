import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, session, redirect, url_for, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils.preprocess import preprocess_image
from functools import wraps
from fpdf import FPDF
from flask_mail import Mail, Message

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:yourpassword@localhost/plant_app"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------------- DATABASE MODELS ----------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Detection(db.Model):
    __tablename__ = "detection"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    leaf_name = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    filename = db.Column(db.String(200))
    moisture = db.Column(db.String(100))
    water = db.Column(db.String(100))
    causes = db.Column(db.Text)
    supplements = db.Column(db.Text)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ---------------- MAIL CONFIG ----------------
app.config.update(
    MAIL_SERVER="smtp.gmail.com",
    MAIL_PORT=465,
    MAIL_USE_SSL=True,
    MAIL_USERNAME="rithwikguruprasad0@gmail.com",
    MAIL_PASSWORD="jhgowtpqyaaaazib",
    MAIL_DEFAULT_SENDER="rithwikguruprasad0@gmail.com"
)
mail = Mail(app)

# ---------------- ML MODELS ----------------
leaf_model = load_model(os.path.join("model", "leaf_detector_final.keras"))
disease_model = load_model(os.path.join("model", "model.keras"))
classes = np.load(os.path.join("model", "label_classes.npy"), allow_pickle=True).tolist()

# ---------------- UTILS ----------------
def load_disease_details(xlsx_path="model/leafs.xlsx"):
    df = pd.read_excel(xlsx_path)
    disease_dict = {}
    for _, row in df.iterrows():
        disease_name = str(row['Leaf Names']).strip()
        disease_dict[disease_name] = {
            "moisture": row.get('Moisture Level'),
            "water": row.get('Water Level'),
            "causes": row.get('Causes'),
            "supplements": row.get('Supplements to Add')
        }
    return disease_dict

DISEASE_DETAILS = load_disease_details()

def is_leaf(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = leaf_model.predict(img_array)[0][0]
    return prediction < 0.2

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index2.html", message=None, success=None)

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    user = User.query.filter_by(username=username).first()
    if not user:
        return render_template("index2.html", message="No username found!", success=False)
    elif not check_password_hash(user.password, password):
        return render_template("index2.html", message="Incorrect password!", success=False)
    session["username"] = user.username
    return redirect(url_for("home"))

@app.route("/signup", methods=["POST"])
def signup():
    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]
    confirm_password = request.form["confirm_password"]
    if password != confirm_password:
        return render_template("index2.html", message="Passwords do not match!", success=False)
    if User.query.filter((User.username==username)|(User.email==email)).first():
        return render_template("index2.html", message="Username or email already exists!", success=False)
    hashed = generate_password_hash(password)
    user = User(username=username, email=email, password=hashed)
    db.session.add(user)
    db.session.commit()
    return render_template("index2.html", message="Account created successfully! Please login.", success=True)

@app.route("/logout")
@login_required
def logout():
    session.pop("username", None)
    return redirect(url_for("index"))

@app.route("/home")
@login_required
def home():
    return render_template("home.html", message=None, success=None)

@app.route("/about")
@login_required
def about():
    return render_template("about.html")

@app.route("/contacts", methods=["GET"])
@login_required
def contacts():
    return render_template("contacts.html", message=None, success=None)

@app.route("/supplements")
@login_required
def supplements():
    return render_template("supplements.html")

@app.route("/contact", methods=["POST"])
@login_required
def contact():
    name = request.form["name"]
    message_body = request.form["message"]
    try:
        msg = Message(
            subject=f"New message from {name}",
            sender=app.config['MAIL_DEFAULT_SENDER'],
            recipients=["rithwiklcw@gmail.com"]
        )
        msg.body = f"Name: {name}\n\nMessage: {message_body}"
        mail.send(msg)
        return render_template("contacts.html", message="Message sent successfully!", success=True)
    except Exception as e:
        print("Error sending mail:", e)
        return render_template("contacts.html", message="Failed to send message. Try again later.", success=False)

@app.route("/getstarted", methods=["GET", "POST"])
@login_required
def getstarted():
    disease_details = None
    result = None
    filename = None
    confidence = None
    message = None
    if request.method == "POST":
        if "leafImage" not in request.files:
            message = "No file uploaded!"
        else:
            file = request.files["leafImage"]
            if file.filename == "":
                message = "No file selected!"
            else:
                ext = os.path.splitext(file.filename)[1]
                unique_filename = f"{uuid.uuid4()}{ext}"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
                file.save(filepath)
                filename = unique_filename
                if not is_leaf(filepath):
                    result = "Not a leaf!"
                else:
                    img_array = preprocess_image(filepath, target_size=(224,224))
                    preds = disease_model.predict(img_array)[0]
                    predicted_index = int(np.argmax(preds))
                    predicted_class = classes[predicted_index]
                    result = predicted_class
                    disease_details = DISEASE_DETAILS.get(predicted_class)
                    confidence = round(float(preds[predicted_index]) * 100, 2)
                    detection = Detection(
                        username=session.get("username"),
                        leaf_name=predicted_class,
                        confidence=confidence,
                        filename=filename,
                        moisture=(disease_details.get("moisture") if disease_details else None),
                        water=(disease_details.get("water") if disease_details else None),
                        causes=(disease_details.get("causes") if disease_details else None),
                        supplements=(disease_details.get("supplements") if disease_details else None),
                        detected_at=datetime.utcnow()
                    )
                    db.session.add(detection)
                    db.session.commit()
    return render_template("getstarted.html", result=result, filename=filename, disease_details=disease_details, confidence=confidence, message=message, success=True if result else False)

@app.route("/download_pdf/<disease_name>/<filename>")
@login_required
def download_pdf(disease_name, filename):
    confidence = request.args.get("confidence")
    details = DISEASE_DETAILS.get(disease_name)
    if not details:
        return "Disease details not found", 404
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Disease Report: {disease_name}", ln=True, align="C")
    pdf.ln(10)
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(img_path):
        pdf.image(img_path, x=55, w=100)
        pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    for k, v in details.items():
        if isinstance(v, str):
            v = v.replace("–", "-")
        pdf.multi_cell(0, 8, f"{k}: {v}")
        pdf.ln(2)
    if confidence:
        pdf.set_font("Arial", "B", 12)
        pdf.ln(5)
        pdf.cell(0, 8, f"Prediction Confidence: {confidence}%", ln=True)
    pdf_file = os.path.join(app.config["UPLOAD_FOLDER"], f"{disease_name}_report.pdf")
    pdf.output(pdf_file)
    return send_file(pdf_file, as_attachment=True)

@app.route("/history")
@login_required
def history():
    username = session.get("username")
    records = Detection.query.filter_by(username=username).order_by(Detection.detected_at.desc()).all()
    return render_template("history.html", records=records)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
