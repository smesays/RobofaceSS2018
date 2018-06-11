import cv2
import sys
cv2.__version__
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cv2.namedWindow('Tracking')

# Set up tracker
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[3]


tracker = cv2.Tracker_create(tracker_type)

# Read video
#video = cv2.VideoCapture("twopeople.avi")
video = cv2.VideoCapture(0)
# Read first frame.
ok, frame = video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
max_area=0
for face in faces:
    area=face[2]*face[3]
    if area>max_area:
        bbox=tuple(face)
    
    (x, y, w, h) = bbox

cv2.rectangle(frame,(x, y), (x + w, y + h),(255,0,0),2)
cv2.imshow('Tracking',frame)    
#Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)
e1 = cv2.getTickCount()
while (video.isOpened()):
    # Read a new frame
    ok, frame = video.read()

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    
    e2 = cv2.getTickCount()
    if (e2 - e1)/ cv2.getTickFrequency()>10:
        cv2.putText(frame, "Re-Initialising Tracking", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,0,0),2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print("Tracker Re-Initialisation")
        print("Number of Detected faces: ",len(faces))
        max_area=0
        for face in faces:
            area=face[2]*face[3]
            if area>max_area:
                max_area=area
                bbox=tuple(face)
        tracker = cv2.Tracker_create(tracker_type)
        ok = tracker.init(frame, bbox)
        e1 = cv2.getTickCount()


    # Draw bounding box
    if ok:
        # Tracking success
        (x, y, w, h) = bbox
        cv2.rectangle(frame,(int(x), int(y)), (int(x) + int(w), int(y) + int(h)),(255,0,0),2)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print("Tracker Re-Initialisation")
        print("Number of Detected faces: ",len(faces))
        max_area=0
        for face in faces:
            area=face[2]*face[3]
            if area>max_area:
                max_area=area
                bbox=tuple(face)
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
