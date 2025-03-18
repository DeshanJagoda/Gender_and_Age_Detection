# Age-Gender-Detector
A Flask-based web app for age and gender detection using OpenCV's DNN and Caffe models. It detects faces, predicts age (8 ranges) and gender, and processes images with PIL, NumPy, and OpenCV.
# FaceInsight - Age and Gender Detection

A web application that uses OpenCV and Caffe models to detect faces and predict age and gender from images.

## Features

- Face detection using OpenCV DNN
- Age prediction (8 age ranges)
- Gender prediction (Male/Female)
- Modern web interface
- Real-time image processing

## Technologies Used

- Python
- OpenCV
- Flask
- Caffe Models
- NumPy
- PIL (Python Imaging Library)

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FaceInsight.git
cd FaceInsight
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
FaceInsight/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── models/            # Caffe models directory
│   ├── age_net.caffemodel
│   ├── age_deploy.prototxt
│   ├── gender_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
├── templates/         # HTML templates
│   └── index.html
└── static/           # Static files
    └── css/
        └── style.css
```

## Usage

1. Upload an image using the web interface
2. The application will detect faces in the image
3. For each detected face, it will predict:
   - Age (in ranges)
   - Gender

## Screenshots

| Interface | Prediction Screen | prediction Screen | prediction Screen | prediction Screen |
|-----------|---------------|--------------|----------------|--------------|
| <img src="screenshot/interface.png" width="200"> | <img src="screenshot/2nd.png" width="200"> | <img src="screenshot/3rd.png" width="200"> | <img src="screenshot/4th.png" width="200"> | <img src="screenshot/5th.png" width="200"> |
