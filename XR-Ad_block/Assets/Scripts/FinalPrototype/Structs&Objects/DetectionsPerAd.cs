/*
    Summary:
    Carries one text-detection result from OCR inference into OCR
    post-processing, along with the ROI snapshot used for word crops.
*/

using Unity.InferenceEngine;
using UnityEngine;

[System.Serializable]
public readonly struct DetectionsPerAd
{
    public readonly TrackedObject trackedObject;
    public readonly Tensor<float> findTextTensor;
    public readonly RenderTexture roiSnapshot;

    public DetectionsPerAd(TrackedObject ad, Tensor<float> tensor, RenderTexture snapshot)
    {
        trackedObject = ad;
        findTextTensor = tensor;
        roiSnapshot = snapshot;
    }
}
