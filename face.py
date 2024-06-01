import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ID for each face
face_id = 0
trackers = []
face_id_map = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Update trackers
    for tracker in trackers:
        success, box = tracker['tracker'].update(frame)
        if success:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            cv2.putText(frame, f'ID: {tracker["id"]}', (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Detect new faces and add them to the tracker
    for (x, y, w, h) in faces:
        # Check if this face is already being tracked
        if not any([tracker for tracker in trackers if
                    x <= tracker['box'][0] + tracker['box'][2] and x + w >= tracker['box'][0] and
                    y <= tracker['box'][1] + tracker['box'][3] and y + h >= tracker['box'][1]]):
            tracker = cv2.TrackerKCF_create()
            box = (x, y, w, h)
            tracker.init(frame, box)
            trackers.append({'tracker': tracker, 'id': face_id, 'box': box})
            face_id += 1

    # Display the resulting frame
    cv2.imshow('Face Detection and Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
