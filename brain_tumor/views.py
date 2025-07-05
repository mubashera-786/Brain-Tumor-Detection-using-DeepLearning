from django.shortcuts import render
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import io
from django.conf import settings

# Load model once
model_path = os.path.join(settings.BASE_DIR,'brain_tumor', 'brain_tumor_detection_model.h5')
model = tf.keras.models.load_model(model_path)
IMG_SIZE = (150, 150)

def predict_image(request):
    prediction = None

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        img = image.load_img(io.BytesIO(uploaded_image.read()), target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        result = model.predict(img_array)[0][0]
        prediction = "Tumor Detected" if result > 0.5 else "No Tumor"

    return render(request, 'predict.html', {'prediction': prediction})



