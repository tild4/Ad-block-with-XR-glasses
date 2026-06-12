/*
    Summary:
    Groups cropped text-region tensors with the tracked object they came
    from before OCR recognition.
*/

using System.Collections.Generic;

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
