using Unity.InferenceEngine;
using UnityEngine;

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
