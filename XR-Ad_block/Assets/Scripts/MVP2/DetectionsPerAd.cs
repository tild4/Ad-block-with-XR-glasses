using Unity.InferenceEngine;
using UnityEngine;

[System.Serializable]
public readonly struct DetectionsPerAd
{
    public readonly TrackedObject trackedObject;
    public readonly Tensor<float> findTextTensor;
    public readonly RenderTexture roiSnapshot; // Frozen ROI crop for text region cropping
    public readonly Rect yoloBounds;           // YOLO bbox at time of snapshot

    public DetectionsPerAd(
        TrackedObject ad,
        Tensor<float> tensor,
        RenderTexture snapshot,
        Rect bounds
    )
    {
        trackedObject = ad;
        findTextTensor = tensor;
        roiSnapshot = snapshot;
        yoloBounds = bounds;
    }
}
