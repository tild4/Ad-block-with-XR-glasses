/*
    Summary:
    Wraps a cropped text-region tensor passed from OCR post-processing to
    text recognition.
*/

using Unity.InferenceEngine;

public readonly struct TextTensor
{
    public readonly Tensor<float> textRegion;

    public TextTensor(Tensor<float> tensor)
    {
        textRegion = tensor;
    }
}
