import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASS_INFO = {
    'Apple___Apple_scab': ('Apple', 'Sick', 'Apply fungicides like captan or myclobutanil.'),
    'Apple___Black_rot': ('Apple', 'Sick', 'Use fungicides (e.g., sulfur) and prune infected branches.'),
    'Apple___Cedar_apple_rust': ('Apple', 'Sick', 'Apply fungicides (e.g., myclobutanil) in spring.'),
    'Apple___healthy': ('Apple', 'Healthy', None),
    'Blueberry___healthy': ('Blueberry', 'Healthy', None),
    'Cherry_(including_sour)___Powdery_mildew': ('Cherry', 'Sick', 'Use sulfur-based fungicides.'),
    'Cherry_(including_sour)___healthy': ('Cherry', 'Healthy', None),
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ('Corn', 'Sick', 'Apply foliar fungicides (e.g., azoxystrobin).'),
    'Corn_(maize)___Common_rust_': ('Corn', 'Sick', 'Use resistant varieties and fungicides.'),
    'Corn_(maize)___Northern_Leaf_Blight': ('Corn', 'Sick', 'Apply fungicides (e.g., propiconazole).'),
    'Corn_(maize)___healthy': ('Corn', 'Healthy', None),
    'Grape___Black_rot': ('Grape', 'Sick', 'Use fungicides (e.g., myclobutanil).'),
    'Grape___Esca_(Black_Measles)': ('Grape', 'Sick', 'No cure; prune affected vines.'),
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ('Grape', 'Sick', 'Apply fungicides like captan.'),
    'Grape___healthy': ('Grape', 'Healthy', None),
    'Orange___Haunglongbing_(Citrus_greening)': ('Orange', 'Sick', 'No cure; remove infected trees.'),
    'Peach___Bacterial_spot': ('Peach', 'Sick', 'Use copper-based bactericides.'),
    'Peach___healthy': ('Peach', 'Healthy', None),
    'Pepper,_bell___Bacterial_spot': ('Bell Pepper', 'Sick', 'Apply copper sprays.'),
    'Pepper,_bell___healthy': ('Bell Pepper', 'Healthy', None),
    'Potato___Early_blight': ('Potato', 'Sick', 'Use fungicides (e.g., chlorothalonil).'),
    'Potato___Late_blight': ('Potato', 'Sick', 'Apply fungicides like mancozeb.'),
    'Potato___healthy': ('Potato', 'Healthy', None),
    'Raspberry___healthy': ('Raspberry', 'Healthy', None),
    'Soybean___healthy': ('Soybean', 'Healthy', None),
    'Squash___Powdery_mildew': ('Squash', 'Sick', 'Use sulfur or potassium bicarbonate.'),
    'Strawberry___Leaf_scorch': ('Strawberry', 'Sick', 'Apply fungicides (e.g., captan).'),
    'Strawberry___healthy': ('Strawberry', 'Healthy', None),
    'Tomato___Bacterial_spot': ('Tomato', 'Sick', 'Use copper-based sprays.'),
    'Tomato___Early_blight': ('Tomato', 'Sick', 'Use fungicides (e.g., chlorothalonil).'),
    'Tomato___Late_blight': ('Tomato', 'Sick', 'Use fungicides like mancozeb.'),
    'Tomato___Leaf_Mold': ('Tomato', 'Sick', 'Improve ventilation and apply fungicides.'),
    'Tomato___Septoria_leaf_spot': ('Tomato', 'Sick', 'Use fungicides (e.g., chlorothalonil).'),
    'Tomato___Spider_mites Two-spotted_spider_mite': ('Tomato', 'Sick', 'Apply miticides or soap.'),
    'Tomato___Target_Spot': ('Tomato', 'Sick', 'Use fungicides (e.g., azoxystrobin).'),
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ('Tomato', 'Sick', 'Control whiteflies with insecticides.'),
    'Tomato___Tomato_mosaic_virus': ('Tomato', 'Sick', 'No cure; remove infected plants.'),
    'Tomato___healthy': ('Tomato', 'Healthy', None)
}

CLASS_NAMES = sorted(list(CLASS_INFO.keys()))

def load_model():
    return tf.keras.models.load_model('crop_model.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64))
    img_array = np.array(img).astype('float32') / 255
    return img_array

def predict_disease(image_path, model):
    img_array = preprocess_image(image_path)
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = []
    
    # TTA: Average over 3 augmented versions + original
    for i, aug_img in enumerate(datagen.flow(img_batch, batch_size=1)):
        pred = model.predict(aug_img, verbose=0)
        predictions.append(pred)
        if i == 2:
            break
    
    predictions.append(model.predict(img_batch, verbose=0))  # Original image
    avg_pred = np.mean(predictions, axis=0)
    class_idx = np.argmax(avg_pred)
    confidence = np.max(avg_pred) * 100
    class_name = CLASS_NAMES[class_idx]
    print(f"Predicted: {class_name} ({confidence:.2f}%)")
    
    leaf_type, health_status, solution = CLASS_INFO[class_name]
    if health_status == 'Healthy':
        return f"{leaf_type}: Healthy ({confidence:.0f}%)"
    else:
        return f"{leaf_type}, Sick, {solution} ({confidence:.0f}%)"