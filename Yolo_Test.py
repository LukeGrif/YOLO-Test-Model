from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11s.pt")

    model.train(
        data="data.yaml",

        # ---- SPEED / SIZE ----
        epochs=30,              # was 60 – this will cut total work in half
        imgsz=640,              # if still slow, change to 512
        batch=8,                # was -1 (auto picked ~3). Try 8; lower if OOM.
        device=0,
        workers=2,

        # ---- AUGMENTATION (light) ----
        mosaic=0.4,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        perspective=0.0005,
        degrees=5.0,
        translate=0.05,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,

        # ---- REGULARIZATION ----
        dropout=0.0,
        # label_smoothing removed (deprecated in your log)

        # ---- OPTIMIZER / LR ----
        optimizer="auto",
        lr0=0.005,
        lrf=0.02,
        momentum=0.937,
        weight_decay=0.0005,

        # ---- TRAINING CONTROL ----
        patience=10,            # early stop sooner
        warmup_epochs=3.0,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        close_mosaic=10,
        val=True,

        cache=False,
        deterministic=False,
        plots=False,            # small speed win / less I/O
    )

if __name__ == "__main__":
    train_model()
