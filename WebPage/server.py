from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import tensorflow.compat.v1 as tf

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = '/Users/parkchanjin/PycharmProjects/ml-work/LungDiseaseKaggle/WebPage/image'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024

model = load_model('./model/model_vgg19.h5')

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False

result = tf.global_variables_initializer()

@app.route("/", methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            img = request.files["image"]

            if img.filename == " ":
                print("Image must have a filename.")
                return redirect(request.url)

            if not allowed_image(img.filename):
                print("That file extension is not allowed")
                return redirect(request.url)

            else:
                filename = secure_filename(img.filename)
                img.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                print("Image saved")

                img = request.files["image"]
                img = image.load_img(img, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                img_data = preprocess_input(x)
                classes = model.predict(img_data)

                global result
                if classes[0][0] == 1:
                    result = 'NORMAL'
                if classes[0][1] == 1:
                    result = 'PNEUMONIA'
                print(result)

                return redirect(request.url)

    return render_template("index.html", result=result)


if __name__ == '__main__':
	app.run(debug=True)