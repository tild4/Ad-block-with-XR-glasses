using UnityEngine;
using Unity.InferenceEngine;
using System.Collections.Generic;

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

public readonly struct TextTensor
{
    public readonly Tensor<float> textRegion;
    public readonly Rect relativeBounds; // DEBUG FOR UI

    public TextTensor(Tensor<float> tensor, Rect bounds)
    {
        textRegion = tensor;
        relativeBounds = bounds;
    }
}

public readonly struct TextTensorsPerAd
{
    public readonly TrackedObject trackedObject;
    public readonly List<TextTensor> textRegions;

    public TextTensorsPerAd(TrackedObject ad, List<TextTensor> texts)
    {
        trackedObject = ad;
        textRegions = texts;
    }
}

public readonly struct TextsPerAd
{
    public readonly TrackedObject trackedObject;
    public readonly List<string> texts;

    public TextsPerAd(TrackedObject ad, List<string> texts)
    {
        trackedObject = ad;
        this.texts = texts;
    }
}