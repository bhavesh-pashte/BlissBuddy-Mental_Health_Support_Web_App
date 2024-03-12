#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install flask
import numpy as np
from tensorflow.keras import models
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from flask import Flask, request, jsonify, render_template


# In[3]:


app = Flask(__name__)

def depression(output):
    if output>=1 and output<=4:
        text = "Minimal Depression"
    elif output>=5 and output<=9:
        text = "Mild Depression"
    elif output>=10 and output<=14:
        text = "Moderate Depression"
    elif output>=15 and output<=19:
        text = "Moderately severe Depression"
    elif output>=20 and output<=27:
        text = "Severe Depressioin"
    return text

def webcam():
    trained_model = models.load_model('trained_vggface.h5', compile=False)
    trained_model.summary()
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}
    detector = MTCNN()
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    black = np.zeros((96,96))
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        # detect faces in the image
        results = detector.detect_faces(frame)
        # extract the bounding box from the first face
        if len(results) == 1: #len(results)==1 if there is a face detected. len ==0 if no face is detected
            try:
                x1, y1, width, height = results[0]['box']
                x2, y2 = x1 + width, y1 + height
                # extract the face
                face = frame[y1:y2, x1:x2]
                #Draw a rectangle around the face
                cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (255, 0, 0), 2)
                # resize pixels to the model size
                cropped_img = cv2.resize(face, (96,96)) 
                cropped_img_expanded = np.expand_dims(cropped_img, axis=0)
                cropped_img_float = cropped_img_expanded.astype(float)
                prediction = trained_model.predict(cropped_img_float)
                print(prediction)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x1+20, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass
        cv2.imshow('Video',frame)
        try:
            cv2.imshow("frame", cropped_img)
        except:
            cv2.imshow("frame", black)
        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()
    text = emotion_dict[maxindex]
    return text

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    data = webcam()
    if data=="Sad" or data=="Neutral":
        return render_template('form2.html')
    else:
        return render_template('homepage2.html')
    
@app.route('/home2')
def home2():
    return render_template('homepage2.html')

@app.route('/form2')
def form2():
    return render_template('form2.html')

@app.route('/score', methods=['POST'])
def score():
    input1 = int(request.form.get('floatingSelect1'))
    input2 = int(request.form.get('floatingSelect2'))
    input3 = int(request.form.get('floatingSelect3'))
    input4 = int(request.form.get('floatingSelect4'))
    input5 = int(request.form.get('floatingSelect5'))
    input6 = int(request.form.get('floatingSelect6'))
    input7 = int(request.form.get('floatingSelect7'))
    input8 = int(request.form.get('floatingSelect8'))
    input9 = int(request.form.get('floatingSelect9'))
    output = input1+input2+input3+input4+input5+input6+input7+input8+input9
    text = depression(output)
    return render_template('index.html', output_text1 = "Your PHQ-9 score is "+str(output), output_text2 = "You have "+text )

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




