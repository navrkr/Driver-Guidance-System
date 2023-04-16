import cv2

# Open the video file
cap = cv2.VideoCapture('2.mov')

# Check if the video file was opened successfully
if not cap.isOpened():
    print('Error: Could not open the video file')
else:
    # Read the first frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print('Error: Could not read a frame from the video file')
    else:
        # Resize the frame
        resized_frame = cv2.resize(frame, (200, 66)) / 255.0

        # Display the resized frame
        cv2.imshow('Resized Frame', resized_frame)
        cv2.waitKey(0)

    # Release the video file
    cap.release()

# Close all windows
cv2.destroyAllWindows()
