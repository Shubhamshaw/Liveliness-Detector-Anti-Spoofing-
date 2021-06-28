# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 10:07:21 2021

@author: Shubham
"""

import numpy as np
import argparse as ag
import cv2
import os
ap=ag.ArgumentParser()
ap.add_argument("-i","--input",required=True,type=str,help="path to the input video")
ap.add_argument("-o","--output",required=True,type=str,help="path to the output directory of cropped faces")
ap.add_argument("-d","--detector",required=True,type=str,help="path to the open cv face detector")
ap.add_argument("-c","--confidence",required=True,type=float,default=0.5,help="minimum probability to filter weak faces")
ap.add_argument("-s","--skip",required=True,type=int,default=16,help="# of frames to skip before applying face detection")
args=vars(ap.parse_args())

print("[INFO] loading face detector...")
protoPath =os.path.sep.join([args["detector"],"deploy.prototxt"])
modelPath =os.path.sep.join([args["detector"],"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(protoPath, modelPath)
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0
while True:
    (grabbed,frame)=vs.read()
    if not grabbed:
        break
    read=read+1
    if read%args["skip"]!=0:
        continue
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    if len(detections)>0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence>args["confidence"]:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            
            
            p=os.path.sep.join([args["output"],"{}.png".format(saved)])
            cv2.imwrite(p,face)
            saved=saved+1
            print("[INFO] saved {} to disk".format(p))
vs.release()
cv2.destroyAllWindows()