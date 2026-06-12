/*
    Summary:
    Carries one YOLO detection through tracking and OCR preparation,
    including the ROI tensor/snapshot owned by downstream OCR stages.
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
    public RenderTexture RoiSnapshot;
}
