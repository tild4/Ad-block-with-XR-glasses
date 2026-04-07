using Unity.InferenceEngine;
using UnityEngine;

[System.Serializable]
public readonly struct DetectionsPerAd
{
    public readonly TrackedObject trackedObject;
    public readonly Tensor<float> findTextTensor;

    public DetectionsPerAd(TrackedObject ad, Tensor<float> tensor)
    {
        trackedObject = ad;
        findTextTensor = tensor;
    }
}
