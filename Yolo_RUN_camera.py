from ultralytics import YOLO
import cv2

def run_webcam():
    # Load your trained model
    model = YOLO(r"C:\Users\Alparslan\Documents\PhD\YOLO\Test\runs\detect\train7\weights\best.pt")

    # Open webcam (0 = default cam, 1 = USB cam)
    cap = cv2.VideoCapture(0)

    # Optional: set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

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
            device=0     # GPU just like your training code
        )

        # Draw boxes
        annotated = results[0].plot()

        # Display it
        cv2.imshow("YOLO11 Webcam Detection", annotated)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
