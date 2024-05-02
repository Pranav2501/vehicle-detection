# Project: Comparision of Vehicle Detection Techniques
# CS 5330: Pattern Recognition & Computer Vision
# Name: Pranav Raj Sowrirajan Balaji
# Date: 20 April 2024
# Description: This script is used to detect moving vehicles in a video stream using Classic Computer Vision techniques.
# Steps: 
# 1. Capture video from the webcam
# 2. Compute absolute difference between current frame and previous frame (Frame Differencing)
# 3. Apply thresholding to eliminate noise 
# 4. Apply dilation to help fill in holes
# 5. Find contours of the threshold image
# 6. Iterate over the contours and draw bounding boxes around it
# 7. Display the original frame with bounding boxes
# 8. Calculate processing time and FPS
# 9. Break the loop if 'q' is pressed
# 10. Release the video capture object and close all OpenCV windows




import cv2
import numpy as np
import time

# Main function for vehicle detection
def main():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    # Initialize variable to hold the previous frame
    prev_frame = None

    # Initialize performance metrics
    start_time = time.time()
    num_frames = 0
    total_processing_time = 0

    while True:
        # Record start time
        frame_start_time = time.time()

        # Read the current frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur the frame
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If this is the first frame then save its value
        if prev_frame is None:
            prev_frame = gray
            continue

        # Compute absolute difference between current frame and previous frame
        frame_delta = cv2.absdiff(prev_frame, gray)

        # Apply thresholding to eliminate noise
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Apply dilation to help fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours of the threshold image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over the contours and draw bounding boxes around it
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # you can adjust this value according to your need
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show the original frame
        cv2.imshow("Frame", frame)

        # Assign the current frame to the previous frame for next iteration
        prev_frame = gray

        # Calculate processing time and increment total processing time
        frame_processing_time = time.time() - frame_start_time
        total_processing_time += frame_processing_time
        num_frames += 1

        # Calculate average processing time and FPS
        avg_processing_time = total_processing_time / num_frames
        fps = num_frames / (time.time() - start_time)

        print(f"Frame {num_frames}: Processing Time = {frame_processing_time:.2f} sec, Average Processing Time = {avg_processing_time:.2f} sec, FPS = {fps:.2f}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

