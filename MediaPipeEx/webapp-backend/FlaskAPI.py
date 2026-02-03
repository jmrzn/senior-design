from flask import Flask, request, send_file
from flask_cors import CORS
import os

from VideoProcessing import processUserVideo

app = Flask(__name__)
CORS(app) 

@app.route('/process-video', methods=['POST'])
def process_video():
    file = request.files['video']
    filePath = os.path.abspath("input_video.mp4")
    file.save(filePath)
    
    outputfile = processUserVideo(filePath)
    print(outputfile)
    
    return send_file(outputfile, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(port=5000)