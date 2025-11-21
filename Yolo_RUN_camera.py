from ultralytics import YOLO
import cv2

# Global flag for exit button
exit_program = False

def click_event(event, x, y, flags, params):
    global exit_program
    # If left mouse click and coordinates inside button
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 110 and 10 <= y <= 60:
            exit_program = True

def run_webcam():
    global exit_program

    # Load your trained model
    model = YOLO(r"./runs/detect/train4/weights/best.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    cv2.namedWindow("YOLO11 Webcam Detection")
    cv2.setMouseCallback("YOLO11 Webcam Detection", click_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Run YOLO prediction
        results = model.predict(
            frame,
            imgsz=640,
            conf=0.4,
            device=0
        )

        # Draw boxes
        annotated = results[0].plot()

        # Draw exit button
        cv2.rectangle(annotated, (10, 10), (110, 60), (0, 0, 255), -1)
        cv2.putText(annotated, "EXIT", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("YOLO11 Webcam Detection", annotated)

        # Quit if button clicked or 'q' pressed
        if exit_program or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
