from ultralytics import YOLO
import traceback

def main():
    weights_path = r"runs\detect\train4\weights\best.pt"
    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)

    try:
        print("Starting ONNX export...")
        result = model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            dynamic=False,
            simplify=False,  # turn off simplify for debugging
            verbose=True,
            nms=True  # IMPORTANT: include NMS so ONNX output is [x1,y1,x2,y2,score,class]
        )
        print("Export result:", result)
    except Exception as e:
        print("EXPORT FAILED WITH EXCEPTION:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
