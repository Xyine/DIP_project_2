import cv2

tracker = cv2.TrackerKCF_create()

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = video.read()
if not ret:
    print("Error: Could not read the frame from the webcam.")
    exit()

# Let the user select the Region of Interest (ROI) to track
print("Select the object to track and press ENTER or SPACE to confirm.")
print("Press 'c' to cancel selection.")
bbox = cv2.selectROI("Tracking", frame, False)

tracker.init(frame, bbox)

cv2.destroyAllWindows()

while True:

    ret, frame = video.read()
    if not ret:
        print("Error: Could not read the frame from the webcam.")
        break

    success, bbox = tracker.update(frame)

    if success:
        # draw the bounding box
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking Failure", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the current frame
    cv2.imshow("KCF Tracker - Webcam", frame)

    # Exit the loop if the user presses the ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
