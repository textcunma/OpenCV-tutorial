'''
flaskを用いてカメラ映像に画像処理を行う
fkask x OpenCV

参考 https://qiita.com/Gyutan/items/1f81afacc7cac0b07526
'''

import cv2
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)       
        # self.run=Segment()      # セグメンテーション
        self.run=ObjectDetection()  # 物体検出
        
    def __del__(self):
        self.video.release()                    
    def get_frame(self):
        _, image = self.video.read()            
        image=self.run(image)                   # 変換
        _, jpeg = cv2.imencode('.jpg', image)   
        return jpeg.tobytes()                  

# セマンティックセグメンテーション
class Segment():
    def __init__(self):
        ModelPath="../ENet/enet-cityscapes/enet-model.net"
        ColorPath="../ENet/enet-cityscapes/enet-colors.txt"

        color = open(ColorPath).read().strip().split("\n")
        color = [np.array(c.split(",")).astype("int") for c in color]
        self.color = np.array(color, dtype="uint8")

        self.net = cv2.dnn.readNet(ModelPath)       

    def __call__(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), swapRB=True, crop=False)
        
        self.net.setInput(blob)     
        output = self.net.forward()  
        classMap = np.argmax(output[0], axis=0) 
        colormask = self.color[classMap]  
    
        colormask = cv2.resize(colormask, (image.shape[1], image.shape[0]),interpolation=cv2.INTER_NEAREST)
        outputimg = ((0.4 * image) + (0.6 * colormask)).astype("uint8")  
        return outputimg

# 物体検出
class ObjectDetection():
    def __init__(self):
        self.confThreshold = 0.25  
        self.nmsThreshold = 0.4     
        modelConfiguration = "../Yolo-Fastest-opencv-dnn/Yolo-Fastest-voc/yolo-fastest-xl.cfg"
        modelWeights = "../Yolo-Fastest-opencv-dnn/Yolo-Fastest-voc/yolo-fastest-xl.weights"
        classesFile = "../Yolo-Fastest-opencv-dnn/voc.names"

        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights) 
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)   

        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

    def __call__(self, img):
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutputsNames(self.net))
        self.postprocess(img, outs)
        return img

    def getOutputsNames(self,net):
        layersNames = net.getLayerNames()
        return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

    def drawPred(self,img,classId, conf, left, top, right, bottom):
        cv2.rectangle(img, (left, top), (right, bottom), (0,0,255), thickness=4)  
        label = '%.2f' % conf

        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(img, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)

    def postprocess(self,img, outs):
        h,w=img.shape[:2]
        classIds = []       
        confidences = []  
        boxes = []       

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)          
                confidence = scores[classId]          
                if confidence > self.confThreshold:        
                    center_x = int(detection[0] * w)  
                    center_y = int(detection[1] * h)  
                    width = int(detection[2] * w)      
                    height = int(detection[3] * h)    
                    left = int(center_x - width / 2)   
                    top = int(center_y - height / 2)   
                    classIds.append(classId)              
                    confidences.append(float(confidence))  
                    boxes.append([left, top, width, height])  
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(img,classIds[i], confidences[i], left, top, left + width, top + height)  