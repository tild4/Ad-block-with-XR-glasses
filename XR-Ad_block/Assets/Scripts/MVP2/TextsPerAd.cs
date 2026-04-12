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
