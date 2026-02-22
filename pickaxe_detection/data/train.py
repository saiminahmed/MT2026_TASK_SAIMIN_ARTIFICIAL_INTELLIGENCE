from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('yolov8s.pt')

    # Train
    model.train(
        data='pickaxe_detection/data/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='pickaxe_detector_v2',
        hsv_h=0.015,   
        hsv_s=0.7,      
        hsv_v=0.4,      
        fliplr=0.5,     
        translate=0.1,  
        scale=0.5,      
    )