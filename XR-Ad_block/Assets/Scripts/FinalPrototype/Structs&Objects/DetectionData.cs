/*
    DetectionData
    Struct to hold data for each detected object, including bounding box, confidence, frame info, and ROI data for OCR.
*/

using Unity.InferenceEngine;
using UnityEngine;

[System.Serializable]
public struct DetectionData
{
    // Normalized bounding box from YOLO (0..1), with top-left origin.
    public Rect bboxNormalized;
    public float confidence;
    public FrameData frame;
    public Tensor<float> RoiTensor;
    public RenderTexture RoiSnapshot; // Persistent copy of the YOLO ROI crop for OCR
    public Rect RoiContentRectNormalized; // Content rect inside RoiSnapshot after aspect padding
}
