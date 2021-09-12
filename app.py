import os
import uuid
from flask import Flask, render_template, request
from analyzer import Analyzer


analyzer = Analyzer(
    binary_classifier_path='models/binary_classifier.cbm',
    vectorizer_path='models/vectorizer.pkl',
    factor_classifier_path='models/factor_classifier.pkl'
)

ALLOWED_EXTENSIONS = {'.docx', '.doc'}
app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'uploads'


def convert_doc2docx(document_path):
    os.system('soffice --headless --convert-to docx --outdir uploads {}'.format(document_path))
    return document_path + 'x'


@app.route('/')
def index():
    return render_template('index.html', corrupt_texts={}.items())


@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    file_extension = os.path.splitext(uploaded_file.filename.lower())[-1]

    if file_extension not in ALLOWED_EXTENSIONS:
        return render_template('index.html', corrupt_texts={}.items())

    filename = str(uuid.uuid4()) + file_extension
    document_path = os.path.join(app.config['UPLOAD_PATH'], filename)
    uploaded_file.save(document_path)

    if os.path.splitext(filename.lower())[-1] == '.doc':
        document_path = convert_doc2docx(document_path)

    corrupt_texts = analyzer.analyze(document_path)

    return render_template('index.html', corrupt_texts=corrupt_texts.items())
