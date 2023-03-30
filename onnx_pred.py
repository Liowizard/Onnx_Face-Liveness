import numpy as np
#import tf2onnx
import onnxruntime as rt
import cv2
faceClassifier = cv2.CascadeClassifier('haarface.xml')
camera = cv2.VideoCapture(0)



def predict(img, model_path = "face_liveness.onnx"):
    if img.shape != (112, 112, 3):
        return -1

    dummy_face = np.expand_dims(np.array(img, dtype=np.float32), axis = 0) / 255.

    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(model_path, providers=providers)
    onnx_pred = m.run(['activation_5'], {"input": dummy_face})
    #print(onnx_pred)
    # print(dummy_face.shape)
    liveness_score = list(onnx_pred[0][0])[1]

    return liveness_score



while cv2.waitKey(1) & 0xFF != ord('q'):
    _, img = camera.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

    for x, y, w, h in faces:
        faceRegion = img[y:y + h, x:x + w]
        x
        faceRegion = cv2.resize(faceRegion, (112, 112)) 

        # print('faceRegion',faceRegion)
        ff1s = predict(faceRegion)
        print(ff1s)
       


        if ff1s < 0.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'Fake', (x, y + h + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        elif ff1s <=0.9:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(img, "Face not clear", (x, y + h + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255))
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'Real', (x, y + h + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

    cv2.imshow('Deep Pixel-wise Binary Supervision Anti-Spoofing', img)