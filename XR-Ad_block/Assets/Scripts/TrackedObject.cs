/*
    TrackedObject

    PURPOSE:
    A persistent data class that represents a unique object (ad) over its entire 
    lifecycle. It maintains identity and state across multiple video frames.

    ARCHITECTURE:
    - Identity: Stores the unique 'id' assigned by the TrackingManager.
    - Persistence: Uses 'timeToLive' (TTL) to keep the object alive even if 
      detections are momentarily lost (grace period).
    - State Management: 
        1. isAnalyzed: Tracks if the object has already been processed by OCR/AI.
        2. text: Stores the recognized content of the ad.
        3. shouldBlock: A boolean flag used by the PlacementManager to 
           decide if a 3D block should be rendered.

    IMPORTANT:
    This is a 'class' (reference type), allowing multiple managers to point 
    to the same object and update its state (e.g., OCR updating the text 
    while TrackingManager updates the position).
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