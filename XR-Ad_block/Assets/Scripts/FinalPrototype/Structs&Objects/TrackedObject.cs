/*
    Summary:
    Stores the live tracking state, OCR result, and block decision for one
    detected object.
*/

[System.Serializable]
public class TrackedObject
{
    public int id;
    public DetectionData lastDetection;
    public float timeToLive;
    public bool isAnalyzed;
    public string text;
    public bool shouldBlock;
}
