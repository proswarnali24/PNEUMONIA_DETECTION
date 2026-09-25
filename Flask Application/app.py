import os
import ssl

# Disable SSL verification for Keras pre-trained weights download on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Set KERAS_HOME inside the project directory
os.environ['KERAS_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.keras'))
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)

import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19


base_model = VGG19(include_top=False, input_shape=(128,128,3))
x = base_model.output
flat=Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)

weights_path = os.path.join(os.path.dirname(__file__), 'vgg_unfrozen.h5')
if os.path.exists(weights_path):
    model_03.load_weights(weights_path)
    print("Loaded model weights from", weights_path)
else:
    print(f"Warning: {weights_path} not found. Running model with initial weights.")

app = Flask(__name__)

print('Model loaded. Flask app running at http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "Normal"
	elif classNo==1:
		return "Pneumonia"
	return "Unknown"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((128, 128))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model_03.predict(input_img)
    result01=np.argmax(result,axis=1)
    return result01[0]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True, port=5000)