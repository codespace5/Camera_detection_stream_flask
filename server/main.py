from flask import Flask, Response
import cv2
import threading
import face_recognition as fr
app = Flask(__name__)

import os
import pickle
import mediapipe
import numpy as np
mp_face_detection = mediapipe.solutions.face_detection
face_detector =  mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence = 0.5)

# initialize a lock used to ensure thread-safe
# exchanges of the frames (useful for multiple browsers/tabs
# are viewing tthe stream)

path = "./train/"
known_names = []
images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

with open('mypickle.pickle' ,'rb') as f:
    loaded_obj = pickle.load(f)
known_name_encodings = loaded_obj
kk  =0
kk1  =0



lock = threading.Lock()
lock1 = threading.Lock()



@app.route('/stream',methods = ['GET'])
def stream():
   return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


def generate():
   # grab global references to the lock variable
   global lock
   global kk
   # initialize the video stream
   vc = cv2.VideoCapture('rtsp://admin:Fourfour@98.113.224.145:80/rtsp/streaming?channel=01&subtype=0')
   # vc = cv2.VideoCapture('test1.mp4')
   # check camera is open
   if vc.isOpened():
      rval, frame = vc.read()
   else:
      rval = False

   # while streaming
   while rval:
      # wait until the lock is acquired
      with lock:
         # read next frame
         rval, frame = vc.read()

         # if blank frame
         if frame is None:
            continue

         # encode the frame in JPEG format

         # image = cv2.flip(image, 1)
         image = frame
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         image= cv2.resize(image, (650, 550))
         if kk< 2:
            results = face_detector.process(image)
            if results.detections:
               for face in results.detections:
                     confidence = face.score
                     bounding_box = face.location_data.relative_bounding_box
                     x = int(bounding_box.xmin * image.shape[1]) - 20
                     w = int(bounding_box.width * image.shape[1]) + 40
                     y = int(bounding_box.ymin * image.shape[0]) -20
                     h = int(bounding_box.height * image.shape[0]) +40

                     image1 = image[y:y+h, x:x+w]
                     image1 = np.ascontiguousarray(image1)
                     face_locations = fr.face_locations(image1)
                     face_encodings = fr.face_encodings(image1, face_locations)
                     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = fr.compare_faces(known_name_encodings, face_encoding)
                        name = "unknown"
                        face_distances = fr.face_distance(known_name_encodings, face_encoding)
                        best_match = np.argmin(face_distances)
                        if matches[best_match]:
                           name = known_names[best_match]
                        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(image, name, (x + 6, y - 6), font, 1.0, (255, 0, 255), 1)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness = 2)
         kk += 1
         if kk> 2:
            kk = 0
         
         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
   
         (flag, encodedImage) = cv2.imencode(".jpg", image)

         # ensure the frame was successfully encoded
         if not flag:
            continue

      # yield the output frame in the byte format
      yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
   # release the camera
   vc.release()


@app.route('/stream1',methods = ['GET'])
def stream1():
   return Response(generate1(), mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate1():
   # grab global references to the lock variable
   global lock1
   global kk1
   # initialize the video stream
   vc = cv2.VideoCapture('rtsp://admin:Fourfour@98.113.224.145:80/rtsp/streaming?channel=02&subtype=0')
   # vc = cv2.VideoCapture('test6.mp4')
   # check camera is open
   if vc.isOpened():
      rval, frame = vc.read()
   else:
      rval = False

   # while streaming
   while rval:
      # wait until the lock is acquired
      with lock1:
         # read next frame
         rval, frame = vc.read()

         # if blank frame
         if frame is None:
            continue

         # encode the frame in JPEG format

         # image = cv2.flip(image, 1)
         image = frame
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         image= cv2.resize(image, (650, 550))
         if kk1< 2:
            results = face_detector.process(image)
            if results.detections:
               for face in results.detections:
                     confidence = face.score
                     bounding_box = face.location_data.relative_bounding_box
                     x = int(bounding_box.xmin * image.shape[1]) - 20
                     w = int(bounding_box.width * image.shape[1]) + 40
                     y = int(bounding_box.ymin * image.shape[0]) -20
                     h = int(bounding_box.height * image.shape[0]) +40

                     image1 = image[y:y+h, x:x+w]
                     image1 = np.ascontiguousarray(image1)
                     face_locations = fr.face_locations(image1)
                     face_encodings = fr.face_encodings(image1, face_locations)
                     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = fr.compare_faces(known_name_encodings, face_encoding)
                        name = "unknown"
                        face_distances = fr.face_distance(known_name_encodings, face_encoding)
                        best_match = np.argmin(face_distances)
                        if matches[best_match]:
                           name = known_names[best_match]
                        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(image, name, (x + 6, y - 6), font, 1.0, (255, 0, 255), 1)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness = 2)
         kk1 += 1
         if kk1> 2:
            kk1 = 0
         
         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
   
         (flag, encodedImage) = cv2.imencode(".jpg", image)

         # ensure the frame was successfully encoded
         if not flag:
            continue

      # yield the output frame in the byte format
      yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
   # release the camera
   vc.release()

if __name__ == '__main__':
   host = "127.0.0.1"
   port = 8000
   debug = False
   app.run(host, port, debug, use_reloader=True, threaded=True)