# importing necessary libraries
import cv2
import datetime
import imutils
import numpy as np
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


def main():
    #getting thee video file
    cap = cv2.VideoCapture('file1.wmv')
    #initializing timer 
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    while True:
        ret, frame = cap.read()
        # resizing the frame size
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
            if confidence > 0.2:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue
                # if person getes dectected make the bounding box
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

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
            text = "ID: {}".format(objectId)
            t= str(time_diff)
            cv2.putText(frame, t, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        # display the output 
        cv2.imshow("Application", frame)
        # print(text,time_diff )
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
    cv2.destroyAllWindows()



