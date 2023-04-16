import tensorflow.compat.v1 as tf   
tf.disable_v2_behavior()
import model
import cv2
import numpy as np
from subprocess import call
import tkinter as tk
import os
import time
from threading import Thread
import pygame
import pygame.mixer as mixer
pygame.init()

# Define the function to perform YOLO object detection
def modelobj(frame):
    # Define labels and colours for the objects to be detected

    labels = [
        'Speed Breaker', 'Bike', 'Car', 'Dog', 'Cattle', 'Potholes', 'Truck', 'Bus', 'Kart', 'Person', 'Auto'
    ]

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="detect.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define the minimum confidence threshold for object detection
    min_confidence = 0.60
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (320, 320))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_confidence) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * 720)))
            xmin = int(max(1,(boxes[i][1] * 1280)))
            ymax = int(min(720,(boxes[i][2] * 720)))
            xmax = int(min(1280,(boxes[i][3] * 1280)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            if(object_name=="Potholes"):
                mixer.music.load('pothole.mp3')
                mixer.music.set_volume(1)
                mixer.music.play()
            elif(object_name=="Speed Breaker"):
                mixer.music.load('sb.mp3')
                mixer.music.set_volume(1)
                mixer.music.play()
            label = object_name+': {:.2f}%'.format(scores[i]*100)  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    #return it detected frame
    return(frame)

def run():
    # Start a timer
    start_time = time.time()
    frame_count = 0
    dir = []
    left_curve_img = cv2.imread('left_turn.png')
    right_curve_img = cv2.imread('right_turn.png')
    keep_straight_img = cv2.imread('straight.png')
    left_curve_img = cv2.normalize(src=left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_curve_img = cv2.normalize(src=right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    keep_straight_img = cv2.normalize(src=keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "save/model.ckpt")

    img = cv2.imread('steering_wheel_image.jpg',0)
    rows,cols = img.shape

    smoothed_angle = 0
    # Video capture
    cap = cv2.VideoCapture(1)
    while(cv2.waitKey(10) != ord('q')):
        ret, frame = cap.read()
        image = cv2.resize(frame, (200, 66)) / 255.0
        degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / 3.14159265
        print("Predicted steering angle: " + str(degrees) + " degrees")
        
        frame = cv2.resize(frame, (1280, 720))

        # Run object detection on the preprocessed frame
        frame = modelobj(frame)

        # Compute the steering advice based on the predicted angle
        if degrees > 7:
            dir.append('R')
        elif degrees < -20:
            dir.append('L')
        else:
            dir.append('F')

        if len(dir) > 9:
            dir.pop(0)

        result = all(element == dir[0] for element in dir)

        if(result):
            if(dir[0]=='R'):
                mixer.music.load('right.mp3')
            elif(dir[0]=='L'):
                mixer.music.load('left.mp3')
            else:
                mixer.music.load('straight.mp3')
            mixer.music.set_volume(1)
            start = time.time()
            mixer.music.play()

        W = 400
        H = 275
        widget = np.copy(frame[:H, :W])
        widget //= 2
        widget[0,:] = [0, 0, 255]
        widget[-1,:] = [0, 0, 255]
        widget[:,0] = [0, 0, 255]
        widget[:,-1] = [0, 0, 255]
        frame[:H, :W] = widget
        
        direction = max(set(dir), key = dir.count)
        
        if direction == 'L':
            y, x = left_curve_img[:,:,2].nonzero()
            frame[y, x-100+W//2] = left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"
        if direction == 'R':
            y, x = right_curve_img[:,:,2].nonzero()
            frame[y, x-100+W//2] = right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"
        if direction == 'F':
            y, x = keep_straight_img[:,:,2].nonzero()
            frame[y, x-100+W//2] = keep_straight_img[y, x, :3]
            msg = "Keep Straight Ahead"

        cv2.putText(frame, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        
        # Calculate and print the FPS on the frame
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("frame", frame)

        #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
        #and the predicted angle
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        cv2.imshow("steering wheel", dst)

    cap.release()
    cv2.destroyAllWindows()

def obj():
    # Start a timer
    start_time = time.time()
    frame_count = 0
    cap = cv2.VideoCapture(1)
    while(cv2.waitKey(10) != ord('q')):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        # Run object detection on the preprocessed frame
        frame = modelobj(frame)

        # Calculate and print the FPS on the frame
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("frame", frame)

    cap.release()
    cv2.destroyAllWindows()   

def path():
    # Start a timer
    start_time = time.time()
    frame_count = 0
    dir = []
    left_curve_img = cv2.imread('left_turn.png')
    right_curve_img = cv2.imread('right_turn.png')
    keep_straight_img = cv2.imread('straight.png')
    left_curve_img = cv2.normalize(src=left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_curve_img = cv2.normalize(src=right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    keep_straight_img = cv2.normalize(src=keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "save/model.ckpt")

    img = cv2.imread('steering_wheel_image.jpg',0)
    rows,cols = img.shape

    smoothed_angle = 0
    # Video capture
    cap = cv2.VideoCapture(1)
    while(cv2.waitKey(10) != ord('q')):
        ret, frame = cap.read()
        image = cv2.resize(frame, (200, 66)) / 255.0
        degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / 3.14159265
        print("Predicted steering angle: " + str(degrees) + " degrees")
        frame = cv2.resize(frame, (1280, 720))
        # Compute the steering advice based on the predicted angle
        if degrees > 7:
            dir.append('R')
        elif degrees < -20:
            dir.append('L')
        else:
            dir.append('F')

        if len(dir) > 9:
            dir.pop(0)

        result = all(element == dir[0] for element in dir)

        if(result):
            if(dir[0]=='R'):
                mixer.music.load('right.mp3')
            elif(dir[0]=='L'):
                mixer.music.load('left.mp3')
            else:
                mixer.music.load('straight.mp3')
            mixer.music.set_volume(1)
            start = time.time()
            mixer.music.play()

        W = 400
        H = 275
        widget = np.copy(frame[:H, :W])
        widget //= 2
        widget[0,:] = [0, 0, 255]
        widget[-1,:] = [0, 0, 255]
        widget[:,0] = [0, 0, 255]
        widget[:,-1] = [0, 0, 255]
        frame[:H, :W] = widget
        
        direction = max(set(dir), key = dir.count)
        
        if direction == 'L':
            y, x = left_curve_img[:,:,2].nonzero()
            frame[y, x-100+W//2] = left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"
        if direction == 'R':
            y, x = right_curve_img[:,:,2].nonzero()
            frame[y, x-100+W//2] = right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"
        if direction == 'F':
            y, x = keep_straight_img[:,:,2].nonzero()
            frame[y, x-100+W//2] = keep_straight_img[y, x, :3]
            msg = "Keep Straight Ahead"

        cv2.putText(frame, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        
        # Calculate and print the FPS on the frame
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("frame", frame)

        #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
        #and the predicted angle
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        cv2.imshow("steering wheel", dst)

    cap.release()
    cv2.destroyAllWindows()   

def toggle_checkbuttons(var, checkbuttons):
    """
    Toggles the state of the given checkbuttons based on the state of the given variable.
    """
    state = var.get()
    for cb in checkbuttons:
        if cb is not var:
            cb.set(state)

# function to run selected scripts
def run_scripts():
    if obj_var.get() == 1 and path_var.get() == 1:
        run()
    elif obj_var.get() == 1:
        obj()
    elif path_var.get() == 1:
        path()
    else:
        print("No scripts selected")
        
# create tkinter window
root = tk.Tk()
root.geometry("720x1080")
root.title("Driver Guidance System")

items = ["obj_var", "path_var"]
listbox = tk.Listbox(root)
for item in items:
    listbox.insert(tk.END, item)

# create a BooleanVar to track the toggle button state
select_all_var = tk.BooleanVar()

# create labels for checkboxes
obj_label = tk.Label(root, text="Object Detection")
path_label = tk.Label(root, text="Path Prediction")
both_label = tk.Label(root, text="Both")

# create checkboxes
obj_var = tk.IntVar()
obj_check = tk.Checkbutton(root, text="", variable=obj_var)
path_var = tk.IntVar()
path_check = tk.Checkbutton(root, text="", variable=path_var)
both_var = tk.IntVar()
both_check = tk.Checkbutton(root, text="", variable=both_var,
                               command=lambda: toggle_checkbuttons(both_var, [obj_var, path_var]))

# add images next to checkboxes
obj_image = tk.PhotoImage(file="object_detection.png")
obj_label2 = tk.Label(image=obj_image)
obj_label2.image = obj_image

path_image = tk.PhotoImage(file="path_prediction.png")
path_label2 = tk.Label(image=path_image)
path_label2.image = path_image

both_image = tk.PhotoImage(file="both.png")
both_label2 = tk.Label(image=both_image)
both_label2.image = both_image


# create button to run selected scripts
run_button = tk.Button(root, text="Run", command=run_scripts)

# place the button at the center of the window
run_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# place all elements in window
obj_check.grid(row=0, column=0, padx=10, pady=10)
obj_label.grid(row=0, column=1, padx=10, pady=10)
obj_label2.grid(row=0, column=2, padx=10, pady=10)
path_check.grid(row=1, column=0, padx=10, pady=10)
path_label.grid(row=1, column=1, padx=10, pady=10)
path_label2.grid(row=1, column=2, padx=10, pady=10)
both_check.grid(row=2, column=0, padx=10, pady=10)
both_label.grid(row=2, column=1, padx=10, pady=10)
both_label2.grid(row=2, column=2, padx=10, pady=10)
run_button.grid(row=3, column=2, padx=10, pady=10)

root.mainloop()