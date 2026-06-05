/*
    TrackedObject

    This class represents an object that is being tracked in the AR environment. It contains:
    - id: A unique identifier for the tracked object.
    - lastDetection: The most recent detection data for this object (YOLO bbox + confidence + frame pose/texture + optional ROI data for OCR).
    - timeToLive: A timer that indicates how long the object should be kept in the tracking system without receiving new detections. This helps to remove objects that are no longer visible or relevant.
    - isAnalyzed: A flag to indicate whether the object has been analyzed for text recognition or other processing.
    - text: The recognized text associated with the object, if any.
    - shouldBlock: A flag to indicate whether a blocking visual should be placed for this object.
*/

using System;
using UnityEngine;

[System.Serializable]
public class TrackedObject
{
    public int id;
    public DetectionData lastDetection; // The most recent detection data for this object.
    public float timeToLive;
    public bool isAnalyzed;
    public string text;
    public bool shouldBlock;
}
