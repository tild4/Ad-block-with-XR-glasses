# Ad-block with XR-glasses

This project aims to detect and block advertisements in real-time using XR technology and the YOLOv8 object detection model.

---

## AI Core (Ad Detection)
This module handles the training and inference of the YOLOv8 model used to identify advertisements.

### Installation
To install the necessary libraries and dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

### Usage

#### Training
To train the model on the current dataset, run:

```bash
python AI-core/train_model.py
```
Note: Training logs and results are saved in the /runs directory (this directory is ignored by Git)

#### Testing
To verify that the model can find advertisements in a test image, run:

```bash
python AI-core/test_model.py
```

## Project Structure (AI-core)

```
AI-core/
│
├── models/        # Contains the latest trained weights (best.pt).
│                  # This is the "brain" used by the application.
│
├── images/        # Training and validation images.
│
├── labels/        # YOLO annotation files.
│
├── test_images/   # Place new images here to test the model.
│
├── data.yaml      # Configuration file telling YOLO where to find the data.
│
├── train_model.py
└── test_model.py
```

---

## Note for Mac Users

The `.DS_Store` files and the `/runs` directory are automatically ignored by Git.

After finishing training, remember to manually copy the `best.pt` file from:

```
runs/detect/train/weights/
```

to:

```
AI-core/models/
```
