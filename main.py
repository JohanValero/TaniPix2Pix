import flask
import flask_cors

import io
import os
#import uuid

from PIL import Image

import tensorflow.compat.v1 as tf
import numpy as np

PATH_MODEL = './Generator'
PATH_IMG   = './ImageSave/'

#Load model tensorflow pre-trained.
modelGenerator = tf.keras.models.load_model('./Generator', False)

app = flask.Flask(__name__, static_url_path = '')
flask_cors.CORS(app)

@app.route('/RecibirImagen', methods = ["GET", "POST"])
def ProcessImage():
    #Find the image of the canvas
    file_ = flask.request.files['Image']
    
    #Convert the RGBA image to RGB image
    file_ = Image.open(file_.stream)
    image_ = Image.new("RGB", file_.size, (255, 255, 255))
    image_.paste(file_, mask = file_.split()[3] )

    #Convert image to input tensor
    image_ = np.array(image_.resize((512, 512)))
    image_ = np.asarray(image_/127.5 - 1)
    image_ = tf.convert_to_tensor(image_)
    image_ = tf.expand_dims(image_, 0)

    #Generate image predict and convert in image
    out_image = modelGenerator(image_, training = True)[0]
    out_image = (out_image + 1)*127.5
    out_image = np.uint8(out_image)
    
    #Convert image to blob
    out_image = Image.fromarray(out_image)
    img_io = io.BytesIO()
    out_image.save(img_io, 'PNG')
    
    #Save image in the server side.
    #image_uuid = uuid.uuid4().hex
    #filename_canvas = image_uuid + '_canvas.png'
    #filename_model  = image_uuid + '_model.png'
    #file_.save(os.path.join(PATH_IMG, filename_canvas))
    #out_image.save(os.path.join(PATH_IMG, filename_model))

    #Response with the image
    img_io.seek(0)

    return flask.send_file(
        img_io,
        as_attachment=True,
        attachment_filename = 'prediction.png',
        mimetype = 'image/png'
    )

@app.route('/', methods = ["GET"])
def GetCanvas():
    return flask.render_template("index.html")

if __name__ == "__main__":
    app.run(host = '127.0.0.1', port = 8080, debug = True)