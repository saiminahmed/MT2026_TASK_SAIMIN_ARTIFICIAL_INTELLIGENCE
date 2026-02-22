from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/pickaxe_detector_v2/weights/best.pt')

    results = model.predict(
        source=r'path of the test image',
        save=True,
        conf=0.5
    )