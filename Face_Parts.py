import cv2
import numpy as np
import dlib

webcam = True

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def createBox(img,points,scale=5,masked=False,cropped = True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[points],(255,255,255))
        img = cv2.bitwise_and(img,mask)
        # cv2.imshow('Mask',img)

    if cropped:
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox
        imgCrop = img[y:y+h,x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
        return imgCrop
    else:
        return mask

while True:
    #webcam
    if webcam: success,img = cap.read()
    else:img = cv2.imread('2.jpg')

    # image
    # img = cv2.imread('12.jpg')

    img = cv2.resize(img,(0,0),None,1,1)
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        # imgOriginal = cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
        landmarks = predictor(imgGray,face)
        myPoints =[]
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
            cv2.circle(imgOriginal,(x,y),5,(50,50,255),cv2.FILLED)
            cv2.putText(imgOriginal,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.9,(0,0,255),1)

        myPoints = np.array(myPoints)

        imgLefteyebrows = createBox(img,myPoints[17:22],1)
        cv2.imshow('Lefteyebrows',imgLefteyebrows)

        imgRighteyebrows = createBox(img,myPoints[22:27],1)
        cv2.imshow('Righteyebrows',imgRighteyebrows)

        nose = createBox(img,myPoints[27:36],1)
        cv2.imshow('nose',nose)

        imgLeftEye = createBox(img,myPoints[36:42],1)
        cv2.imshow('LeftEye',imgLeftEye)

        imgRightEye = createBox(img,myPoints[42:48],1)
        cv2.imshow('RightEye',imgRightEye)

        imgLips = createBox(img,myPoints[48:61],1)
        cv2.imshow('imgLips',imgLips)

        print(myPoints)

    cv2.imshow("Original",imgOriginal)
    cv2.waitKey(1)
