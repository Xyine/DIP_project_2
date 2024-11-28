import cv2

tracker = cv2.TrackerKCF_create()

video = cv2.VideoCapture("sample_video2.mp4")

# The frames might be bigger than the screen
frame_width = 640
frame_height = 480

# Read the first frame
ret, frame = video.read()
if not ret:
    print("Error: Could not read the video file.")
    exit()

frame = cv2.resize(frame, (frame_width, frame_height))

# Select ROI (Region of Interest) for tracking
print("Select the object to track and press ENTER or SPACE to confirm.")
bbox = cv2.selectROI("Tracking", frame, False)

tracker.init(frame, bbox)

cv2.destroyAllWindows()

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    success, bbox = tracker.update(frame)

    if success:
        # Draw bounding box
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking Failure", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("KCF Tracker", frame)

    # Exit on ESC key
    if cv2.waitKey(30) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
