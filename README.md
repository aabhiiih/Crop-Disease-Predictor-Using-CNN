# Crop-Disease-Predictor-Using-CNN

A machine learning-based web application to detect crop diseases from leaf images, providing predictions and treatment suggestions. Built with TensorFlow, Streamlit, and MySQL, it achieves 79% accuracy on the PlantVillage dataset (54,000+ images, 38 classes). Ideal for farmers to promote early disease detection and sustainable agriculture.

## Features

* Disease Prediction: Upload leaf images to classify 38 crop disease classes across 10 crop types.

* User Authentication: Secure login and signup with MySQL database.

* Crop History: Track uploaded images and predictions per user.

* EDA Visualization: View prediction distribution via Matplotlib plots.

* Treatment Suggestions: Actionable solutions for diseased crops.

## Installation

* Prerequisites

* Python 3.8+

* MySQL Server

* Git


## Usage

* Login/Signup: Create an account or log in to access the app.

* Upload Image: Upload a leaf image (JPG/PNG) to predict disease.

* View Results: Get instant predictions (e.g., "Tomato: Healthy" or "Apple, Sick, Apply fungicides").

* Check History: View past uploads and predictions.

* Explore EDA: See a plot of your prediction distribution.


## Dataset

* Source: PlantVillage Dataset(https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

* Details: 54,000+ RGB images, 38 classes (e.g., Apple___Apple_scab, Tomato___healthy).

* Preprocessing: Resized to 64x64, normalized, augmented (rotation, flip, zoom).

## Model Details

* Architecture: Custom CNN (4 Conv2D layers, BatchNormalization, Dropout).

* Training: 80-20 train-validation split, 8 epochs, Adam optimizer.

* Performance: ~93% validation accuracy.

* Prediction: Uses Test-Time Augmentation (TTA) for robust results.

## Requirements

See requirements.txt for details. Key libraries:

* streamlit

* tensorflow

* mysql-connector-python

* pandas

* matplotlib

* numpy

* pillow

* sqlalchemy


## Limitations

* Trained on 64x64 images, potentially missing fine details.

* Requires stable internet for Streamlit deployment.

* Model may misclassify diseases with similar visual symptoms.

## Future Enhancements

* Train on higher-resolution images (e.g., 128x128).

* Develop a mobile app for field use.

* Integrate real-time weather data for contextual predictions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

* PlantVillage for the dataset.

* TensorFlow and Streamlit communities for excellent documentation.

Contact

For questions, reach out to abhinandabhi1001@gmail.com or open an issue on GitHub.
