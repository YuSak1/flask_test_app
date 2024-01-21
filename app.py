import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename#ファイル名をチェックする関数
from keras.models import Sequential,load_model
from PIL import Image
import keras,sys
import numpy as np

#ラベル名、識別するクラス数、リサイズ後の画像サイズ指定
classes = ["man","woman"]
num_classes = len(classes)
image_size = 50


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    #ファイル名に拡張子が存在するか,判定ALLOWED_EXTENSIONSで定義した拡張子であるかを判定
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

#TOPにルーティングした際の処理
@app.route('/', methods=['GET','POST'])#GETとPOSTのみ受け取る
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            #ファイルを識別器に渡して答えを返す。学習済みの識別器は.h5が拡張子
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            model = load_model('./model_weight/man_woman_cnn.h5')

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)#Xをリスト型からnumpyの型に変換

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = float(result[predicted] * 100)

            resultmsg = "ラベル： " + classes[predicted] + ", 確率："+ str(percentage) + " %"
            return render_template('kekka.html', resultmsg=resultmsg, filepath=filepath)

    return render_template('index.html')


from flask import send_from_directory

#画像を受け取ってからの処理
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

#おまじない
if __name__ == '__main__':
    app.run()
