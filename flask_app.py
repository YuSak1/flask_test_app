# import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from keras.models import load_model
from PIL import Image
# import keras
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO


classes = ["man","woman"]
num_classes = len(classes)
image_size = 80

UPLOAD_FOLDER = './saved_images/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('File not found')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('File not found')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            # file.save(os.path.join(UPLOAD_FOLDER,filename))
            # filepath = os.path.join(UPLOAD_FOLDER,filename)
            model = load_model('model_weight/man_woman_cnn_v3.h5', compile=False)

            # image = Image.open(filepath)
            image_original = Image.open(file)
            image = image_original.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            pred = model.predict([X])[0][0]
            # predicted = result.argmax()
            # percentage = float(result[predicted] * 100)
            if pred < 0.5:
              result = "Man"
              percentage = (1 - pred)*100
            else:
              result = "Woman"
              percentage = pred*100

            # resultmsg =  str(percentage) + " %   " + classes[predicted]
            # resultmsg =  '{:.4} %'.format(percentage) + classes[predicted]
            resultmsg =  str(percentage) + " %   " + result

            # os.remove(filepath)
            # print("Image is deleted.")

            # Create feature map images
            model_visualize = load_model('model_weight/man_woman_cnn_v3_visualize.h5', compile=False)
            features = model_visualize.predict([X])

            # for i in range(0,16):
            #     path_img = './saved_images/images/feature_img_' + str(i) + '.jpg'
            #     cmap = plt.get_cmap('Spectral')
            #     img_feature = cmap(features[0,:,:,i], bytes=True)
            #     img_feature = Image.fromarray(img_feature).convert('RGB')
            #     img_feature = img_feature.resize([320,320], resample=Image.NEAREST)
            #     img.save(path_img)

            fig = plt.figure(figsize=(20,10))
            for i in range(1,32+1):
              plt.subplot(4,8,i)
              plt.imshow(features[0,:,:,i-1] , cmap='Spectral')
            img_feature = fig

            #process image for showing
            buf = BytesIO()
            image_original.save(buf,format="png")  
            img_b64str = base64.b64encode(buf.getvalue()).decode("utf-8") 
            img_b64data = "data:image/png;base64,{}".format(img_b64str) 

            #feature image
            with BytesIO() as f:
                plt.savefig(f, format='jpg')
                buf = f.getvalue()

            feature_b64str = base64.b64encode(buf).decode("utf-8") 
            feature_b64data = "data:image/png;base64,{}".format(feature_b64str) 

            return render_template('result.html', resultmsg=resultmsg, face_img=img_b64data, feature_img=feature_b64data)

    return render_template('index.html')

# from flask import send_from_directory

# @app.route('/saved_images/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


#run
if __name__ == '__main__':
    app.run()
