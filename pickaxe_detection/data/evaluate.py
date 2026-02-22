from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/pickaxe_detector_v2/weights/best.pt')

    model.val(
        data='pickaxe_detection/data/data.yaml',
        split='test'
    )