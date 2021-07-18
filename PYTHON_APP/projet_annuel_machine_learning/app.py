from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import os
from ModelApplication import getPredict

app = Flask(__name__)
app.secret_key = "issou"


@app.route('/')
def index():
    modelnames = ['MLP', 'Linear', 'RBF', 'SVM Kernel', 'SVM']
    return render_template('index.html', modelnames=modelnames)


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join('image', 'photoTest.png'))
            label = getPredict(request.form.get("modelnames"))
            print(label)
            flash(label)
            flash(filename)
            return redirect('/')


if __name__ == "__main__":
    app.run()
