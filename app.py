from __future__ import division, print_function
import os
import numpy as np

# Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model_path ='tomato_disease_inception.h5'

# Load your trained model
model = load_model(model_path)

def model_predict(img_path, model):
    
    img = load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!

    pred = model.predict(x)
    pred_binary=int(np.argmax(pred, axis=1))
    if pred_binary == 0:
        result = "Bacterial spot"  
    elif pred_binary == 1:
        result = "Early blight"
    elif pred_binary == 2:
        result = "Healthy"
    elif pred_binary == 3:
        result = "Late blight"
    elif pred_binary == 4:
        result = "Leaf Mold"
    elif pred_binary == 5:
        result = "Septoria leaf spot"
    elif pred_binary == 6:
        result = "Spider_mites"
    elif pred_binary == 7:
        result = "Target Spot"
    elif pred_binary == 8:
        result = "Tomato mosaic virus"
    else:
        result = "Tomato Yellow Leaf Curl Virus" 
    
    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = model_predict(file_path, model)
        return prediction
    return None


if __name__ == '__main__':
    app.run()
