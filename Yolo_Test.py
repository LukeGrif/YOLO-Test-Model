from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11l.pt")    # use LARGE model for small datasets

    model.train(
        data="data.yaml",
        epochs=200,
        imgsz=640,
        batch=32,
        device=0,
        workers=0,

        # ----------------------
        # AUGMENTATION SETTINGS
        # ----------------------
        mosaic=1.0,              # turn on mosaic augmentation
        mixup=0.2,               # mix images
        copy_paste=0.3,          # paste objects between images
        hsv_h=0.015,             # color aug (Hue)
        hsv_s=0.7,               # Saturation
        hsv_v=0.4,               # Value (brightness)
        perspective=0.0005,      # slight perspective warp
        degrees=10,              # random rotation
        translate=0.1,           # shifting
        scale=0.5,               # zoom out
        shear=2.0,               # shear transform
        flipud=0.1,              # vertical flip
        fliplr=0.5,              # horizontal flip

        # ----------------------
        # REGULARIZATION
        # ----------------------
        dropout=0.0,
        label_smoothing=0.1,

        # ----------------------
        # OPTIMIZER / LR CONTROL
        # ----------------------
        optimizer="AdamW",       # very good for small datasets
        lr0=0.001,               # starting learning rate
        lrf=0.01,                # final LR fraction
        momentum=0.937,
        weight_decay=0.0005,

        # ----------------------
        # TRAINING SETTINGS
        # ----------------------
        patience=50,             # early stopping for small datasets
        warmup_epochs=3.0,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        close_mosaic=10,         # disable mosaic in last X epochs
        val=True,                # run validation

        # optional but good
        cache=False,
        deterministic=True,
    )

if __name__ == "__main__":
    train_model()
