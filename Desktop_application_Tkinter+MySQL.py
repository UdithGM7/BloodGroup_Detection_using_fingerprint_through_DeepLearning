import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
import mysql.connector as sql
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pyttsx3
from datetime import datetime
from reportlab.pdfgen import canvas

report_list = []

'''ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")'''

# Initialize pyttsx3
engine = pyttsx3.init()

# Initialize UI
root = ctk.CTk()
root.title("BLOOD GROUP DETECTION")
root.geometry("800x750")

# Database Connection
try:
    connection = sql.connect(host="localhost", user="Reports", password="123123", database="reports")
    cursor = connection.cursor()
except Exception as e:
    messagebox.showerror("Database Error", f"Error connecting to database: {e}")

# Load Model
model_path = "C:/Users/udith/model.h5"
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    messagebox.showerror("Model Error", f"Error loading model: {e}")
    model = None

# Extract Class Names
dataset_path = "C:/Users/udith/Documents/SOFTEN PROJECT/dataset/dataset"
class_names = sorted(os.listdir(dataset_path)) if os.path.exists(dataset_path) else []

# Function to Predict Blood Group
def predict_blood_group(image_path):
    if model is None:
        messagebox.showerror("Error", "Model not loaded.")
        return "Unknown"
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"

# Insert Data into Database
def insert_into_db(name, age, gender, blood_group):
    try:
        cursor.execute("SELECT * FROM reports.blood_test_info WHERE Name = %s", (name,))
        if cursor.fetchone():
            messagebox.showwarning("Duplicate Entry", "Name already exists!")
        else:
            cursor.execute("INSERT INTO reports.blood_test_info (Name, Age, Gender, Blood_Group) VALUES (%s, %s, %s, %s)", 
                           (name, age, gender, blood_group))
            connection.commit()
            messagebox.showinfo("Success", "Details added to database!")
    except Exception as e:
        messagebox.showerror("Database Error", f"An error occurred: {e}")

# Upload Image Function
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.png;*.jpg;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((220, 220))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        predict_button.configure(state="normal", command=lambda: predict_image(file_path))

# Prediction Function
def predict_image(image_path):
    name, age, gender = name_entry.get(), age_entry.get(), gender_var.get()
    if not name or not age or not gender:
        messagebox.showwarning("Input Error", "Please fill all fields before predicting.")
        return
    blood_group = predict_blood_group(image_path)
    result_label.configure(text=f"Predicted Blood Group: {blood_group}")
    insert_into_db(name, age, gender, blood_group)

    report_list.append(name)
    report_list.append(age)
    report_list.append(gender)
    report_list.append(blood_group)

    engine.say(f"The predicted blood group is {blood_group}")
    engine.runAndWait()

# Save Report as PDF
def save_report_as_pdf():
    folder_path = filedialog.askdirectory(title="Select Folder to Save Report")  
    file_path = os.path.join(folder_path, f"{report_list[0]}_BloodTestReport.pdf")
    c = canvas.Canvas(file_path)

    c.setFont("Helvetica-Bold", 20)
    c.drawString(200, 750, "BLOOD TEST REPORT")

    test_date = datetime.now().strftime("%d-%m-%Y")
    c.setFont("Helvetica", 12)
    c.drawString(200, 720, f"Test Date: {test_date}")

    test_time = datetime.now().strftime("%H:%M:%S")
    c.setFont("Helvetica", 12)
    c.drawString(350, 720, f"Test Time: {test_time}")

    c.setFont("Helvetica", 12)
    c.drawString(100, 680, f"Name: {report_list[0]}")
    c.drawString(100, 650, f"Age: {report_list[1]}")
    c.drawString(100, 620, f"Gender: {report_list[2]}")
    c.drawString(100, 590, f"Blood Group: {report_list[3]}")

    c.save()
    messagebox.showinfo("Success", f"Report saved as {file_path}")

# UI Layout
frame = ctk.CTkFrame(root, width=600, height=500,fg_color="#696969")
frame.pack(pady=40, padx=20, fill="both", expand=True)

# Title Label
title_label = ctk.CTkLabel(frame, text="BLOOD GROUP DETECTION", font=("sans serif", 28, "bold"))
title_label.pack(pady=20)

# Input Fields with Labels
input_frame = ctk.CTkFrame(frame)
input_frame.pack(pady=10)

ctk.CTkLabel(input_frame, text="Name:", font=("sans serif", 14)).grid(row=0, column=0, padx=10, pady=5, sticky="w")
name_entry = ctk.CTkEntry(input_frame, width=200)
name_entry.grid(row=0, column=1, padx=10, pady=5)

ctk.CTkLabel(input_frame, text="Age:", font=("sans serif", 14)).grid(row=1, column=0, padx=10, pady=5, sticky="w")
age_entry = ctk.CTkEntry(input_frame, width=200)
age_entry.grid(row=1, column=1, padx=10, pady=5)

ctk.CTkLabel(input_frame, text="Gender:", font=("sans serif", 14)).grid(row=2, column=0, padx=10, pady=5, sticky="w")
gender_var = ctk.StringVar()
gender_dropdown = ctk.CTkComboBox(input_frame, variable=gender_var, values=["Male", "Female", "Other"], width=200)
gender_dropdown.grid(row=2, column=1, padx=10, pady=5)

# Image Upload Button
upload_button = ctk.CTkButton(frame, text="Upload Image",command=upload_image, width=200,fg_color="#3D3D3D")
upload_button.pack(pady=10)

# Image Preview
image_label = ctk.CTkLabel(frame, text="No Image Uploaded", width=200, height=200, fg_color="gray")
image_label.pack(pady=10)

# Predict Button
predict_button = ctk.CTkButton(frame, text="Predict", state="disabled", width=200,fg_color="#3D3D3D")
predict_button.pack(pady=10)

# Result Label
result_label = ctk.CTkLabel(frame, text="Prediction Result", font=("sans serif", 16, "bold"))
result_label.pack(pady=15)

# Save Report Button
save_button = ctk.CTkButton(frame, text="Save Report", command=save_report_as_pdf, width=200,fg_color="#3D3D3D")
save_button.pack(pady=10)

report_list.clear()
root.mainloop()
