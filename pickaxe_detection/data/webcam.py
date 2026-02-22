from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/pickaxe_detector_v2/weights/best.pt')

    model.predict(
        source=0,        # 0 = default webcam
        show=True,       # displays the live window
        conf=0.25,        # confidence threshold
        save=False       # set True if you want to save the footage
    )