#!/usr/bin/env python

# Copyright (c) 2011, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Willow Garage, Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import rospy
from geometry_msgs.msg import Twist
import sys, select, os
import cv2
import numpy as np
from ultralytics import YOLO

if os.name == 'nt':
    import msvcrt
else:
    import tty, termios

ROBOT_MAX_LIN_VEL = 3.0
ROBOT_MAX_ANG_VEL = 2.84

ROBOT_LINEAR_VEL_STEP_SIZE = 0.1
ROBOT_ANGULAR_VEL_STEP_SIZE = 0.02

robot_node_name = 'robot_teleop'

msg = """
Control your robot!
-------------------
Moving key
        w
    a   s   d
        x 
w/x     : increase/decrease linear velocity
a/d     : increase/decrease angular velocity
s/space : stop robot
Press Ctrl+C to quit
"""

err = """
Communications failed -_-
"""


def getKey():
    if os.name == 'nt':
        return msvcrt.getch();

    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key 

def vels(target_linear_vel, target_angular_vel):
    return "currently:\tlinear vel %s\tangular vel %s " % (target_linear_vel, target_angular_vel)

def makeSimpleProfile(output, input, slop):
    if input > output:
        output = min(input, output + slop)
    elif input < output:
        output = max(input, output - slop)
    else:
        output = input
    return output

def constrain(input, low, high):
    if input < low:
        input = low
    elif input > high:
        input = high
    else:
        input = input
    return input

def checkLinearLimitVelocity(vel):
    vel = constrain(vel, -ROBOT_MAX_LIN_VEL, ROBOT_MAX_LIN_VEL)
    return vel 

def checkAngularLimitVelocity(vel):
    vel = constrain(vel, -ROBOT_MAX_ANG_VEL, ROBOT_MAX_ANG_VEL)
    return vel

def boundingboxPredict(image):
   try:
      target_linear_vel, target_angular_vel = 2.0, 0.0
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
            target_linear_vel = 0.0
            target_angular_vel = 0.0
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
               target_angular_vel = 0.0
            if offset < -20 and offset > -120:
               cv2.putText(img_detect, "Left move -10 deg" ,(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
               target_angular_vel = -10.0
            elif offset < -120 and offset > -220:
               cv2.putText(img_detect, "Left move -20 deg",(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
               target_angular_vel = -20.0
            elif offset < -220 and offset > -320:
               cv2.putText(img_detect, "Left move -30 deg",(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
               target_angular_vel = -30.0
            
            if offset > 20 and offset < 120:
               cv2.putText(img_detect, "Right move 10 deg" ,(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
               target_angular_vel = 10.0
            elif offset > 120 and offset < 220:
               cv2.putText(img_detect, "Right move 20 deg",(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
               target_angular_vel = 20.0
            elif offset > 220 and offset < 320:
               cv2.putText(img_detect, "Right move 30 deg",(int(image_width/2),30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
               target_angular_vel = 30.0
   except:
      pass
   target_angular_vel = target_angular_vel*3.14/180
   return img_detect, class_name, target_linear_vel, target_angular_vel

if __name__ == "__main__":
    model_detect = YOLO("yolov8n.pt")
# model_seg = YOLO("weights_segment_parking_40_epochs.pt")

    cap = cv2.VideoCapture(0)
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rospy.init_node(robot_node_name)
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    target_linear_vel   = 0.0
    target_angular_vel  = 0.0


    control_linear_vel  = 0.0
    control_angular_vel = 0.0

    try:
        print(msg)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                image, class_name, target_linear_vel, target_angular_vel = boundingboxPredict(frame)
                target_linear_vel = checkLinearLimitVelocity(target_linear_vel)
                target_angular_vel = checkAngularLimitVelocity(target_angular_vel)
                cv2.imshow('Person Detection', image)
                cv2.waitKey(1)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break

            #target_linear_vel = checkLinearLimitVelocity(target_linear_vel)
            #print(vels(target_linear_vel, target_angular_vel))
            
            #target_linear_vel = checkLinearLimitVelocity(target_linear_vel)
            #print(vels(target_linear_vel, target_angular_vel))
            
            #target_angular_vel = checkAngularLimitVelocity(target_angular_vel)
            #print(vels(target_linear_vel, target_angular_vel))
            
            #target_angular_vel = checkAngularLimitVelocity(target_angular_vel)
            #print(vels(target_linear_vel, target_angular_vel))
            
            #target_linear_vel   = 0.0
            #control_linear_vel  = 0.0
            #target_angular_vel  = 0.0
            #control_angular_vel = 0.0
            #print(vels(target_linear_vel, target_angular_vel))

            key = getKey()
            if key == '\x03':
                break



            twist = Twist()

            control_linear_vel = makeSimpleProfile(control_linear_vel, target_linear_vel, (ROBOT_LINEAR_VEL_STEP_SIZE/2.0))
            twist.linear.x = control_linear_vel
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            control_angular_vel = makeSimpleProfile(control_angular_vel, target_angular_vel, (ROBOT_ANGULAR_VEL_STEP_SIZE/2.0))
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = control_angular_vel

            pub.publish(twist)

    except:
        print(err)

    finally:
        twist = Twist()
        twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
        pub.publish(twist)

    if os.name != 'nt':
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
