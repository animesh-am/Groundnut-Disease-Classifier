from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = None  # Initialize model (loaded later)
class_names = ['Alternaria Leafspot', 'Early Late Leafspot', 'Healthy',
               'Rosette', 'Rust']


def load_custom_model():
    global model
    if model is None:
        model = load_model('MobileNet102-0.9767.hdf5')
        model.make_predict_function()


def predict_label(img_path):
    load_custom_model()
    try:
        print("Image Path:", img_path)
        print("Class Names:", class_names)

        # Load and preprocess the image
        test_image = image.load_img(img_path, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)

        print("Image Shape:", test_image.shape)

        # Make predictions
        predictions = model.predict(test_image)
        print("Raw Predictions:", predictions)

        # Post-process the predictions
        result = predictions.flatten()
        print(result)

        index = result.argmax()
        confidence = round(result[index] * 100, 2)
        pred_class = class_names[index]

        return pred_class, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


@app.route("/", methods = ['GET', 'POST'])
def main():
    if request.method == 'POST':
        img = request.files['image']
        img_path = "static/images" + img.filename
        img.save(img_path)

        predicted_class, confidence = predict_label(img_path)
        return render_template("index.html", prediction = predicted_class, confidence = confidence, img_path = img_path)

    return render_template('index.html', prediction = None, confidence = None, img_path = None)


# if __name__ == '__main__':
#     os.makedirs("static", exist_ok = True)
#     app.run(debug = True)