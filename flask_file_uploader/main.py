import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
import uuid
import scan_video
	
@app.route('/')
def home():
	return  'Deepfake Detection App'

@app.route('/', methods=['POST'])
def upload_video():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
        #create a unique file name
		filename = filename = str(uuid.uuid4()) + ".mp4"
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		result = scan_video.detect(filename='videos/' + filename)
		print(result)

		return jsonify(result=result)

if __name__ == "__main__":
    app.run(port=3005)