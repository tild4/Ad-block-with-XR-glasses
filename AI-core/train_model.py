from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# epochs how many times to go through the entire dataset during training
# imgsz is the size to which all images will be resized during training (640x640 in this case)
model.train(data='AI-core/data.yaml', epochs=100, imgsz=640)