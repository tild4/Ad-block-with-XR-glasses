from ultralytics import YOLO
import cv2

# load the trained model (replace with the path to your best.pt file)
model = YOLO('AI-core/models/best.pt')

# run prediction on a completely new image
results = model('AI-core/test_images/IMG_7891.jpg')

for r in results:
    # returns an RGB image array with the detections drawn on it
    image_array = r.plot()

    cv2.imshow('YOLO Test - Did we find advertising?', image_array)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()