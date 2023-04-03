#!/usr/bin/env python

import rospy
from ultralytics import YOLO
import cv2
import numpy as np


def boundingboxPredict(image):
   try:
      img_detect = cv2.resize(np.copy(image), dsize=(640,480))
      image_size = img_detect.shape
      image_width = image_size[1]
      image_height = image_size[0]
      cv2.putText(img_detect, "({},{})".format(image_width,image_height),(10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
                  
      result_detect = model_detect.predict(img_detect)
      area_list = []


      for out in result_detect:
            boxes = out.boxes.xyxy
            classes = out.boxes.cls 
            
            for i, class_name in enumerate(classes):
                  if class_name == 0:
                     class_name = "Person"
                     x1, y1, x2, y2 = boxes[i]
                     x1 = int(x1)
                     y1 = int(y1)
                     x2 = int(x2)
                     y2 = int(y2)
                     
                     
                     area = (x2 - x1)*(y2 - y1)
                     area_list.append([area,x1,y1,x2,y2])

      area_list.sort()
      print(area_list[-1])
      print("-----------------------------")
      x1 = area_list[-1][1]
      y1 = area_list[-1][2]
      x2 = area_list[-1][3]
      y2 = area_list[-1][4]
      if area_list[-1][0] >= 100000:
            cv2.putText(img_detect, "STOPPPPPP!", (int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            
      elif area_list[-1][0] < 100000:
            
            cv2.rectangle(img_detect, (x1, y1), (x2, y2), (0, 0, 255), 2)
            x_obj_center = int((x1+x2)/2)
            y_obj_center = int((y1+y2)/2)
            #  center = (x_obj_center, y_obj_center)
            cv2.line(img_detect, (x_obj_center,y1), (x_obj_center,y2), (0,0,255), 2)
            cv2.line(img_detect, (x1,y_obj_center), (x2,y_obj_center), (0,0,255), 2)
            cv2.putText(img_detect, "Person", (x1+3, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            cv2.putText(img_detect, "({},{})".format(x_obj_center,y_obj_center),(x1,y_obj_center-20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            

            
            cv2.putText(img_detect, "Area={} pixels".format(area),(20,image_height - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            
            offset = x_obj_center - image_width/2
            
            if offset > -20 and offset < 20:
               cv2.putText(img_detect, "Stop 0 deg" ,(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)

            if offset < -20 and offset > -120:
               cv2.putText(img_detect, "Left move -10 deg" ,(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
            elif offset < -120 and offset > -220:
               cv2.putText(img_detect, "Left move -20 deg",(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
            elif offset < -220 and offset > -320:
               cv2.putText(img_detect, "Left move -30 deg",(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
            
            if offset > 20 and offset < 120:
               cv2.putText(img_detect, "Right move 10 deg" ,(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
            elif offset > 120 and offset < 220:
               cv2.putText(img_detect, "Right move 20 deg",(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
            elif offset > 220 and offset < 320:
               cv2.putText(img_detect, "Right move 30 deg",(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
   except:
      pass

   return img_detect, class_name

model_detect = YOLO("yolov8n.pt")
# model_seg = YOLO("weights_segment_parking_40_epochs.pt")

cap = cv2.VideoCapture(0)

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:

    image, class_name = boundingboxPredict(frame)
    cv2.imshow('Person Detection', image)
    cv2.waitKey(1)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else: 
    break
cap.release()
cv2.destroyAllWindows()



# #Segmentation


# results_seg = model_seg(cv_image)

# for result in results_seg:
#     out_seg_ = np.zeros(shape= (480, 640))
#     for i, name_class in enumerate(result.boxes.cls):
#         if name_class == 0:
#             out_seg = (result.masks.masks[i].cpu().numpy()*255).astype('uint8')


#             out_seg_ += out_seg

# cv2.imshow(..)