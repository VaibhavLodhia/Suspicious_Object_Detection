
import numpy as np
import cv2
from collections import Counter, defaultdict
import datetime
import imutils
import numpy as np
from numpy.lib.function_base import append
from centroidtracker import CentroidTracker

#initializing Modelnet ssd(DNN module allows loading pre-trained models of most popular deep learning frameworks, including Tensorflow, Caffe, Darknet, Torch)
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        # initialize the list of picked indexes
        pick = []
        
	#  coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
    # computing the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # coordinates(x,y)for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # computing the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # computing the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
            overlap = (w * h) / area[idxs[:last]]
            # deleting all indexes from the index list that are in the suppression list
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))
# location of first frame


firstframe = cv2.imread('C:/Users/Dell/Desktop/Resoluteai/task4/abandoned-object-detection/FrameNo0.png')
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray,(21,21),0)



cap = cv2.VideoCapture('C:/Users/Dell/Desktop/Resoluteai/task4/abandoned-object-detection/video1.avi')

consecutiveframe=20

track_temp=[]
track_master=[]
track_temp2=[]

top_contour_dict = defaultdict(int)
obj_detected_dict = defaultdict(int)
# img = cv2.imread('C:/Users/Dell/Desktop/Resoluteai/task4/abandoned-object-detection/ri.jpg')
# logo = cv2.resize(img, (250,100))
disp = []
person = []
# Create a mask of logo
# img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frameno=0
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    
    # logo = cv2.imread('logo.png')

    
    if ret == 0:
        continue

    frame = imutils.resize(frame, width=640)
    total_frames = total_frames + 1

    (H, W) = frame.shape[:2]
        #calling blobFromImages function as there is less function call overhead and you'll be able to batch process the images/frames faster.
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        # set the input to the pre-trained deep learning network and obtain
        # the output 
        # class label predictions
    detector.setInput(blob)
    person_detections = detector.forward()
    rects = []
        #detecting the person
    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])
                 
            if CLASSES[idx] != "person":
                continue
                # if person getes dectected make the bounding box
            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")
            rects.append(person_box)
            if rects!=[]:
                person.append("Person Detected")
                
    boundingboxes = np.array(rects)
    boundingboxes = boundingboxes.astype(int)
        # calling non_max_suppression_fast to select the bounding box
    rects = non_max_suppression_fast(boundingboxes, 0.3)
    objects = tracker.update(rects)

            
        # timer end
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
        # calculatinfg frames/sec
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)
        
        # Deawing the box 
    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

            
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #gives Id of the person/object
        # text = "ID: {}".format(objectId)
        # t= str(time_diff)
        # cv2.putText(frame, "person", (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        # cv2.putText(frame,str(person), (640,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        
    # cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    frameno = frameno + 1
    # cv2.putText(frame,'%s%.f'%('Frameno:',frameno), (400,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray,(21,21),0)
     
    frame_diff = cv2.absdiff(firstframe, frame)
  
    #Canny Edge Detection
    edged = cv2.Canny(frame_diff,10,200) #any gradient between 30 and 150 are considered edges
    # cv2.imshow('CannyEdgeDet',edged)
    kernel2 = np.ones((5,5),np.uint8) #higher the kernel, eg (10,10), more will be eroded or dilated
    thresh2 = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel2,iterations=2)
    # cv2.imshow('main', frame)
    
    #Create a copy of the thresh to find contours    
    cnts, _ = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mycnts =[] # every new frame, set to empty list. 
    # loop over the contours
    for c in cnts:


        # Calculate Centroid using cv2.moments
        M = cv2.moments(c)
        if M['m00'] == 0: 
            pass
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])



            
            if cv2.contourArea(c) < 200 or cv2.contourArea(c)>20000:
                pass
            else:
                mycnts.append(c)
                  
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.putText(frame,'C %s,%s,%.0f'%(cx,cy,cx+cy), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2) 
                
                
                #Store the cx+cy, a single value into a list ; max length of 10000
                #Once hit 10000, tranfer top 20 points to dictionary ; empty list
                sumcxcy=cx+cy
                
                
                
                #track_list.append(cx+cy)
                track_temp.append([cx+cy,frameno])
                
                
                track_master.append([cx+cy,frameno])
                countuniqueframe = set(j for i, j in track_master) # get a set of unique frameno. then len(countuniqueframe)
                

                if len(countuniqueframe)>consecutiveframe or False: 
                    minframeno=min(j for i, j in track_master)
                    for i, j in track_master:
                        if j != minframeno: # get a new list. omit the those with the minframeno
                            track_temp2.append([i,j])
                
                    track_master=list(track_temp2) # transfer to the master list
                    track_temp2=[]
                    
      
                
                countcxcy = Counter(i for i, j in track_master)
           
                for i,j in countcxcy.items(): 
                    if j>=consecutiveframe:
                        top_contour_dict[i] += 1
  
                
                if sumcxcy in top_contour_dict:
                    if top_contour_dict[sumcxcy]>100:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        cv2.putText(frame,'%s'%('CheckObject'), (cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                   
                        disp.append(sumcxcy)

                        obj_detected_dict[sumcxcy]=frameno
    
    for i, j in obj_detected_dict.items():
        if frameno - obj_detected_dict[i]>200:
            print ('PopBefore',i, obj_detected_dict[i],frameno,obj_detected_dict)
            print ('PopBefore : top_contour :',top_contour_dict)
            obj_detected_dict.pop(i)
            
            # Set the count for eg 448 to zero. because it has not be 'activated' for 200 frames. Likely, to have been removed.
            top_contour_dict[i]=0
            print ('PopAfter',i, obj_detected_dict[i],frameno,obj_detected_dict)
            print ('PopAfter : top_contour :',top_contour_dict)

                        
    black_area = np.zeros([480, 400, 3], dtype=np.uint8)
    black_area.fill(175)
    frame = np.concatenate((frame, black_area), axis=1)
    # frame[0:logo.shape[0], frame.shape[1] - logo.shape[1] -100 :frame.shape[1]-100] = logo
    # left = np.zeros((frame.shape[0],250, frame.shape[2]), dtype=frame.dtype)
    # cv2.putText(left, "Some information", (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    # cv2.putText(left, "More information", (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    # conv=''
    text1 = map(str, obj_detected_dict) 
    # t=''.join(map(str, text1))
    
    # t2 = ''.join(map(str, text2))
    # text = str(conv.join(text1))
    # print(type(text1))
    for i in disp:
        if disp!=[]:
            # print(disp[0])
            # disp.append(504)
            t1 = ''.join([str(disp[0])])
            # print(disp[1])
        t = ''.join(t1)
        
        cv2.putText(frame,"Location:"+ t +","+t, (640,280), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    time = datetime.datetime.now().strftime("%H:%M:%S")   
    cv2.putText(frame,"Live Stats:", (640,150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame,"Date:"+" "+"Jun 29", (640,170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(frame,"Time:"+" "+str(time), (640,190), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(frame,"Area:" +" "+ "Corridor", (640,210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(frame,"----------------------------------------------------", (640,225), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    cv2.putText(frame,"Suspicious Object at:", (640,250), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(frame,str(person), (640,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    cv2.putText(frame,"----------------------------------------------------", (640,300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    cv2.putText(frame,"People Detection Stats", (640,320), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    if person!=[]:
        cv2.putText(frame,"Person Detected", (640,350), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
   
    
    cv2.imshow('main',frame)


    
         
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()
