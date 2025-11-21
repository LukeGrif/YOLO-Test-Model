import sys
from ultralytics import YOLO
import cv2

def run_single_image(img_path):
    # Path to your trained model
    model = YOLO(r"C:\Users\Alparslan\Documents\PhD\YOLO\Test\runs\detect\train7\weights\best.pt")

    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not load image → {img_path}")
        return

    # Run YOLO11 prediction
    results = model.predict(
        img,
        imgsz=640,
        conf=0.48,
        device=0
    )

    # Draw bounding boxes
    annotated_img = results[0].plot()

    # Display
    cv2.imshow("YOLO11 - Single Image Detection", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional save
    save_path = r"C:\Users\Alparslan\Documents\PhD\YOLO\Test\single_prediction.jpg"
    cv2.imwrite(save_path, annotated_img)
    print(f"Saved result to: {save_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    run_single_image(img_path)
