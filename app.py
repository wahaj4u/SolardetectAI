import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import visualkeras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import random

# Function to load the pre-trained VGG16 model
def load_trained_model():
    # Load the custom trained model
    model = tf.keras.models.load_model('my_trained_model.keras')
    
    return model

# Function to provide recommendations based on the predicted class
def provide_recommendations(predicted_class):
    recommendations = {
        0: "The solar panel is dusty and may experience reduced efficiency. It's recommended to clean the panel to restore its optimal performance. Use a soft cloth or a gentle air blower to remove dust without damaging the surface. Avoid using harsh chemicals or abrasives.",
        1: "The solar panel is in good condition and clean. No immediate action is needed. Keep monitoring regularly to ensure it remains clean and efficient.",
        2: "There appears to be electrical damage on the solar panel. This can lead to performance issues or complete failure. It is important to consult a technician to assess and repair the damaged wiring, inverter, or other electrical components.",
        3: "The solar panel has bird droppings on it, which can affect its efficiency. Clean the panel carefully using a damp cloth. If the droppings have caused staining, you may need to use a mild detergent designed for solar panel cleaning.",
        4: "The solar panel has physical damage such as cracks or dents. This could significantly reduce the panel's efficiency and lifespan. It is advisable to have the panel inspected and potentially replaced to ensure safe and efficient energy generation.",
        5: "The solar panel is covered with snow. Snow accumulation can block sunlight and decrease energy production. It is recommended to clear the snow gently using a soft brush or rake designed for solar panel cleaning. Be cautious to avoid scratching the surface."
    }
    return recommendations.get(predicted_class, "Unknown condition. Please inspect the panel.")

# Function to classify the uploaded image
def classify_image(model, img):
    img = img.resize((244, 244))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Function to visualize thermal images
def generate_thermal_image(img):
    img_array = np.array(img)
    thermal_image = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)  # Apply thermal colormap
    return thermal_image

# Function to simulate defect segmentation (simple thresholding example)
def segment_defects(img):
    img_array = np.array(img)
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)  # Simple threshold
    return thresholded

def send_email_notification(predicted_class, recommendation):
    sender_email = "wahajulislam6@gmail.com"
    receiver_email = "markjonesigi@gmail.com"
    subject = f"Solar Panel Defect Alert: {class_names[predicted_class]}"
    
    # Corrected f-string formatting for the email body
    body = f"""Alert: The solar panel condition has been detected as {class_names[predicted_class]}.

Recommendation:
{provide_recommendations(predicted_class)}
"""

# Setting up the MIME structure
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect to the Gmail server and send the email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, "mmajpvcywdvjyeav")
            server.send_message(msg)
        st.success(f"Email successfully sent to {receiver_email}!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# Streamlit Interface
st.title("Solar Panel Image Classification & Defect Detection")
st.write("Upload an image of a solar panel to classify it, generate a thermal image, and detect any hidden defects.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)
    
    # Load model
    model = load_trained_model()


    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Classify the image
    st.write("Classifying the image...")
    predicted_class = classify_image(model, image)
    class_names = ["Dusty", "Clean", "Electrical Damage", "Bird-dropping", "Physical Damage", "Snow Covered"]
    st.write(f"Predicted Class: {class_names[predicted_class]}")
    
    # Generate thermal image
    st.write("Generating Thermal Image...")
    thermal_image = generate_thermal_image(image)
    st.image(thermal_image, caption="Generated Thermal Image", use_container_width=True)

    # Segment defects (simulated with thresholding)
    st.write("Segmenting Defects...")
    segmented_image = segment_defects(image)
    st.image(segmented_image, caption="Segmented Defects", use_container_width=True)

    # Visualize the model structure
    st.write("Visualizing the Model Structure")
    # Generate the model plot using visualkeras
    model_plot = visualkeras.layered_view(model, legend=True, spacing=50, background_fill='white')
    
    # Recommendations based on the predicted class
    predicted_class = classify_image(model, image)
    class_names = ["Dusty", "Clean", "Electrical Damage", "Bird-dropping", "Physical Damage", "Snow Covered"]
    st.write(f"Predicted Class: {class_names[predicted_class]}")
    st.write(f"Recommendation: {provide_recommendations(predicted_class)}")

    # Trigger email notification if defect detected
    if predicted_class != "clean":  
        send_email_notification(predicted_class, provide_recommendations(predicted_class))

    

