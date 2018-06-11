
# coding: utf-8

# In[ ]:


import cv2
import sys
cv2.__version__
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

cv2.namedWindow('Tracking')

# Set up tracker
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[4]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

# Read video
video = cv2.VideoCapture("twopeople.avi")


# Read first frame.
ok, frame = video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for face in faces:
    bbox = tuple(face)
    print(bbox)
(x, y, w, h) = bbox
cv2.rectangle(frame,(x, y), (x + w, y + h),(255,0,0),2)
cv2.imshow('Tracking',frame)    

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while (video.isOpened()):
    # Read a new frame
    ok, frame = video.read()

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        (x, y, w, h) = bbox
        cv2.rectangle(frame,(int(x), int(y)), (int(x) + int(w), int(y) + int(h)),(255,0,0),2)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        ok, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(len(faces))
        for face in faces:
            bbox = tuple(face)
            #tracker.clear()
            tracker = cv2.Tracker_create(tracker_type)
            ok = tracker.init(frame, bbox)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    
    # Display result
    cv2.imshow("Tracking", frame)
    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):        
        break
cv2.destroyAllWindows()            
video.release()

