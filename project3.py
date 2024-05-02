# Project: Comparision of Vehicle Detection Techniques
# CS 5330: Pattern Recognition & Computer Vision
# Name: Pranav Raj Sowrirajan Balaji
# Date: 20 April 2024
# Description: This script is used to detect moving vehicles in a video stream using YOLOv8 model.


import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


# Main function for vehicle detection
def main():
    # Load the YOLOv8 model
    model = YOLO("yolov8n-seg.pt")  # segmentation model

    # Load the classes
    names = model.model.names

    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    # Get the frame width, height and FPS
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Initialize the video writer
    out = cv2.VideoWriter('instance-segmentation.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    # Process the video frame by frame
    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        # Perform the forward pass
        results = model.predict(im0)
        annotator = Annotator(im0, line_width=2)

        # Draw the bounding boxes and masks
        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            for mask, cls in zip(masks, clss):
                annotator.seg_bbox(mask=mask,
                                mask_color=colors(int(cls), True),
                                det_label=names[int(cls)])

        # Save the annotated frame
        out.write(im0)
        # Display the annotated frame
        cv2.imshow("instance-segmentation", im0)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video writer, video capture object and close all OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
