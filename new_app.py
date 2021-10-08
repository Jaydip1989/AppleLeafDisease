from flask import Flask, render_template, redirect, url_for, request
import os
import numpy as np
import tensorflow
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array

model = keras.models.load_model('AppleLeafDiseaseClassifier.hdf5')

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def uploads():
    classes = ['Black rot', 'Cedar rust', 'Healthy', 'Scab']
    if request.method == "POST":
        f = request.files['imagefile']
        basepath = os.path.dirname(__name__)
        filename = f.filename
        image_path = os.path.join(
            basepath, 'static/uploads', filename
        )
        f.save(image_path)

        IMG = load_img(image_path, target_size=(299, 299))
        IMG = img_to_array(IMG)
        img = IMG/255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)

        label = np.argmax(prediction, axis=1)
        prediction_c = classes[label[0]]

        predicted_class = prediction_c.lower()

        return render_template('base.html', filename=filename, prediction=predicted_class)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
