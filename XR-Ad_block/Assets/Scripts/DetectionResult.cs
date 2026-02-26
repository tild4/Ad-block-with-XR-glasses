using System;
using UnityEngine;

[Serializable]
public struct DetectionResult
{
    public Vector2 viewportPoint; // normalized 0�1
    public float confidence;
    public string label;
    public DateTime timestamp;
}
