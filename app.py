from flask import Flask, render_template, request, flash, redirect
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from flask_ngrok3 import run_with_ngrok
from werkzeug.utils import secure_filename

app = Flask(__name__)

run_with_ngrok(app)

model = None  # Initialize model (loaded later)
class_names = ['Alternaria Leafspot', 'Early Late Leafspot', 'Healthy',
               'Rosette', 'Rust']

app.config['UPLOAD_FOLDER'] = 'static/images/'
# Allow only specific file types
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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
        if 'image' not in request.files:
            flash('No image selected')
            return redirect(request.url)

        img = request.files['image']

        if img.filename == '':
            flash('No image selected')
            return redirect(request.url)

        if img and allowed_file(img.filename):
            # Secure the filename to prevent malicious attacks
            filename = secure_filename(img.filename)

            # Save the file only if it's selected
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(img_path)

            predicted_class, confidence = predict_label(img_path)
            return render_template('index.html', prediction = predicted_class, confidence = confidence, img_path = img_path)

    return render_template('index.html', prediction = None, confidence = None, img_path = None)


# Route for each disease page
@app.route("/disease/<disease>")
def disease_page(disease):
    # Render the corresponding disease page
    return render_template(f'disease_{disease}.html')


if __name__ == '__main__':
    os.makedirs("static", exist_ok = True)
    app.run()
