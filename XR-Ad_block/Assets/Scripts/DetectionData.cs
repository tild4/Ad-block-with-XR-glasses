/*
    DetectionData

    PURPOSE:
    A lightweight data container (struct) that represents a single AI detection
    after it has been processed. It holds all the necessary info to place
    a block in the 3D world.

    ARCHITECTURE:
    - Data-only: No logic, just properties (Value Object).
    - Dual Coordinate System:
        1. bboxNormalized: Used for Raycasting (0.0 to 1.0).
        2. bboxPixels: Used for UI/Debug overlays on the screen.
    - Metadata: Stores confidence score and a reference to the source FrameData.

    IMPORTANT:
    Being a 'struct' instead of a 'class' means it's passed by value.
    This is memory-efficient for high-frequency updates (e.g., 30-60 AI detections per second).
*/

using Unity.InferenceEngine;
using UnityEngine;

[System.Serializable]
public struct DetectionData
{
    public Rect bboxNormalized; // 0-1 normalized coordinates

    // Bounding box as (x1, y1, x2, y2) in the same normalized YOLO coordinate space as bboxNormalized.
    // NOTE: YOLO uses top-left origin; converting to Unity viewport requires flipping Y.
    public Vector4 bboxMinMaxNormalized;
    public Rect bboxPixels; // Pixel coordinates
    public float confidence;
    public FrameData frame;
    public Tensor<float> RoiTensor;
}
