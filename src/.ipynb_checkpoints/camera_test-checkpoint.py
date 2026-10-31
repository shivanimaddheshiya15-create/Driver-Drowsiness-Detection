import cv2

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read camera frame
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected!")
        break

    # Display video
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()