from ultralytics import YOLO #for object detection
import cv2 #for video feed handling 


model = YOLO("yolov8n.pt")

#initiate camera 
cap = cv2.VideoCapture(0)

while True :
    ret, frame=cap.read()

    if not ret:
        break

    results = model(frame)

    # Render Detection
    annotated_frame = results[0].plot()

    #Show output
    cv2.imshow("Detection", annotated_frame)

    #exit
    if cv2.waitKey(1) & 0xff == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()