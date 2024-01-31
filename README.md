# Groundnut Disease Classifier

## Overview

This is a simple web application that classifies groundnut diseases based on images uploaded by users. It uses a machine learning model to make predictions about the type of disease affecting the groundnut plant.

## Features

- **Image Upload:** Users can upload images of groundnut plants to the application.
- **Disease Prediction:** The application predicts the type of disease based on the uploaded image.
- **Confidence Level:** Users are provided with a confidence level for each prediction.

## Technologies Used

- **Flask:** Used for building the backend server and handling HTTP requests.
- **HTML/CSS:** Frontend interface for user interaction.
- **Deep Learning Model:** The model uses (MobileNet v2) architecture for disease classification.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/groundnut-disease-classifier.git
   cd groundnut-disease-classifier

## Install Dependencies
   ```bash
   pip install -r requirements.txt

## 2. Run the application
   ```bash
   python app.py
   
## 3. Open in browser
   Open your web browser and go to http://localhost:5000 to access the application.

## Contributing
If you'd like to contribute to the project, please follow the standard GitHub flow:

Fork the repository.
Create a new branch: git checkout -b feature/your-feature.
Make your changes and commit them: git commit -m 'Add your feature'.
Push to the branch: git push origin feature/your-feature.
Submit a pull request.
