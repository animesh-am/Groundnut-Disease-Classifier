# from flask import Flask, render_template, request
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# import os
#
# app = Flask(__name__)
#
# model = None  # Initialize model (loaded later)
# class_names = ['Alternaria Leafspot', 'Early Late Leafspot', 'Healthy',
#                'Rosette', 'Rust']
#
#
# def load_custom_model():
#     global model
#     if model is None:
#         model = load_model('MobileNet102-0.9767.hdf5')
#         model.make_predict_function()
#
#
# def predict_label(img_path):
#     load_custom_model()
#     try:
#         print("Image Path:", img_path)
#         print("Class Names:", class_names)
#
#         # Load and preprocess the image
#         test_image = image.load_img(img_path, target_size = (224, 224))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis = 0)
#
#         print("Image Shape:", test_image.shape)
#
#         # Make predictions
#         predictions = model.predict(test_image)
#         print("Raw Predictions:", predictions)
#
#         # Post-process the predictions
#         result = predictions.flatten()
#         print(result)
#
#         index = result.argmax()
#         confidence = round(result[index] * 100, 2)
#         pred_class = class_names[index]
#
#         return pred_class, confidence
#
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         raise
#
#
# @app.route("/", methods = ['GET', 'POST'])
# def main():
#     if request.method == 'POST':
#         img = request.files['image']
#         img_path = "static/images" + img.filename
#         img.save(img_path)
#
#         predicted_class, confidence = predict_label(img_path)
#         return render_template("index.html", prediction = predicted_class, confidence = confidence, img_path = img_path)
#
#     return render_template('index.html', prediction = None, confidence = None, img_path = None)
#
#
# # if __name__ == '__main__':
# #     os.makedirs("static", exist_ok = True)
# #     app.run(debug = True)
#

import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

model = None
class_names = ['Alternaria Leafspot', 'Early Late Leafspot', 'Healthy', 'Rosette', 'Rust']


def load_custom_model():
    global model
    if model is None:
        model = load_model('MobileNet102-0.9767.hdf5')
        model.make_predict_function()


def predict_label(img_path):
    load_custom_model()
    try:
        st.write("Image Path:", img_path)
        st.write("Class Names:", class_names)

        # Load and preprocess the image
        test_image = image.load_img(img_path, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)

        st.write("Image Shape:", test_image.shape)

        # Make predictions
        predictions = model.predict(test_image)
        st.write("Raw Predictions:", predictions)

        # Post-process the predictions
        result = predictions.flatten()
        st.write(result)

        index = result.argmax()
        confidence = round(result[index] * 100, 2)
        pred_class = class_names[index]

        return pred_class, confidence

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        raise


def main():
    st.title('GROUNDNUT Disease Classifier')

    uploaded_file = st.file_uploader("Choose an image", type = ["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption = "Uploaded Image", use_column_width = True)
        st.write("")
        st.write("Classifying...")

        # Process the image directly from the uploaded file without saving it
        content = uploaded_file.getvalue()
        img_path = process_uploaded_file(content)

        predicted_class, confidence = predict_label(img_path)

        st.success(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence}%")
        st.image(img_path, caption = "Classified Image", use_column_width = True)


def process_uploaded_file(content):
    # Create a temporary file to write the uploaded content
    with st.tempfile.NamedTemporaryFile(delete = False, suffix = ".jpg") as temp_file:
        temp_file.write(content)

    return temp_file.name


if __name__ == '__main__':
    main()
