import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import io
import base64

# Configuration
UPLOAD_FOLDER = './UPLOAD_FOLDER'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Models
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Model configuration
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Custom filter for base64 encoding
@app.template_filter('b64encode')
def b64encode(data):
    return base64.b64encode(data).decode('utf-8')

# Helper function for face detection
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2, 8)
    return frameOpencvDnn, faceBoxes

# Frame generator for uploaded images
def process_uploaded_image(img_file):
    frame = cv2.cvtColor(img_file, cv2.COLOR_RGB2BGR)
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    age_gender_data = []
    
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]):min(faceBox[3], frame.shape[0]-1), 
                     max(0, faceBox[0]):min(faceBox[2], frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]
        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]
        age_gender_data.append({'gender': gender, 'age': age})
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    ret, encodedImg = cv2.imencode('.jpg', resultImg)
    return bytearray(encodedImg), age_gender_data

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['fileToUpload'].read()
        img = Image.open(io.BytesIO(f))
        img_np = np.asarray(img, dtype="uint8")
        processed_img, age_gender_data = process_uploaded_image(img_np)

        # Prepare data to display
        age_gender_display = [f"Gender: {data['gender']}, Age: {data['age']}" for data in age_gender_data]
        
        return render_template('index.html', processed_image=processed_img, age_gender_data=age_gender_display)

if __name__ == '__main__':
    app.run(debug=True)
