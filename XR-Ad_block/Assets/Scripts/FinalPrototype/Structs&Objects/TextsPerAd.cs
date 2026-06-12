/*
    Summary:
    Groups recognized OCR strings with the tracked object they belong to
    before final block/no-block classification.
*/

using System.Collections.Generic;

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
