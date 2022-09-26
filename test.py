import torch
import numpy as np
import cv2
import serial
from Ax12 import Ax12
import time



class MugDetection:
    def __init__(self, capture_index, model_name):
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.ser = serial.Serial('/dev/ttyUSB1', 9600)
        print("Using Device: ",self.device)

    def get_video_capture(self):
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('data/', 'custom', path=model_name, force_reload=True, source='local')
            model.conf = 0.3
            model.iou = 0.25

        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame=[frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:,-1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            # print(row)

            if row[4] >= 0.3:
                x1,y1,x2,y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                center_point = round((x1+x2)/2), round((y1+y2)/2)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]),(x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                cv2.circle(frame, center_point,5,bgr,2)
                cv2.putText(frame,str(center_point), center_point,cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255))
                print(center_point, self.class_to_label(labels[i]))
                self.ser_avr(labels,i)


        return frame

    def ser_avr(self,labels,i):
        if self.class_to_label(labels[i]) == 'welchs':
            data = "0"
            data = data.encode("utf-8")
            self.ser.write(data)
            print(data)
        elif self.class_to_label(labels[i]) == 'paper':
            data = "1"
            data = data.encode("utf-8")
            self.ser.write(data)
            print(data)

        self.ser.close()


    def ser_Ax12(self):
        Ax12.DEVICENAME = '/dev/ttyUSB0'
        Ax12.BAUDRATE = 1_000_000
        Ax12.connect()

        while True:
            data = self.ser.read()
            if data == b'A':
                Ax12(4).set_goal_position(700)
                Ax12(5).set_goal_position(300)
                Ax12(6).set_goal_position(680)
                time.sleep(0.3)
                Ax12(3).set_goal_position(180)
                time.sleep(4)
                Ax12(4).set_goal_position(500)
                Ax12(5).set_goal_position(500)
                Ax12(6).set_goal_position(500)
                time.sleep(1)
                Ax12(4).set_goal_position(400)
                Ax12(5).set_goal_position(600)
                time.sleep(0.3)
                Ax12(6).set_goal_position(350)
                Ax12(3).set_goal_position(50)
                time.sleep(3)
                Ax12(4).set_goal_position(500)
                Ax12(5).set_goal_position(500)
                Ax12(6).set_goal_position(500)
                time.sleep(0.3)
                Ax12(3).set_goal_position(180)



    def __call__(self):
        cap = self.get_video_capture()
        assert  cap.isOpened()

        while True:

            ret, frame = cap.read()
            assert ret

            frame = cv2. resize(frame,(640,640))

            start_time =time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 5/np.round(end_time - start_time, 2)
            #print(f"Frames Per Second : {fps} " )

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow('YOLOV5 Detection', frame)

            if cv2.waitKey(5) & 0xFF ==27:
                break

        cap.release()

detector = MugDetection(capture_index=1, model_name='best2.pt')
detector()

