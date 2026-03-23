/*
    TrackingManager

    PURPOSE:
    The "Brain" of the persistence layer. It ensures that a physical object in the real
    world maintains the same unique ID across multiple camera frames, even if the
    AI detection momentarily flickers or disappears.

    ARCHITECTURE:
    - Object Association: Uses the IOU (Intersection over Union) algorithm to match
      new incoming detections with existing 'TrackedObjects'.
    - ID Management: Assigns a permanent 'nextId' to every new unique detection.
    - Lifecycle Control:
        1. TTL (Time To Live): Keeps objects alive for a few seconds (e.g., 2.0s)
           after they are no longer seen by the AI.
        2. Cleanup: Automatically purges expired objects from the system.
    - OCR Coordination: Identifies new objects that haven't been analyzed yet and
      broadcasts them via 'onNewOCRCandidate'.

    IMPORTANT:
    - The 'iouThreshold' is key: too high and tracking "breaks" easily; too low
      and different objects might swap IDs.
    - 'UpdateDetections' is the main entry point, triggered by the DetectionPostProcessor.
*/

using System;
using System.Collections.Generic;
using UnityEngine;

public class TrackingManager : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private DetectionPostProcessor detectionPostProcessor;

    //[SerializeField] private DecisionManager decisionManager;

    [Header("Tracking Settings")]
    [SerializeField]
    private float timeToLive = 2.0f; // Seconds before object expires

    [SerializeField]
    private float iouThreshold = 0.5f; // Threshold for matching detections

    // List of currently tracked objects
    private List<TrackedObject> trackedObjects = new List<TrackedObject>();

    // ID counter for new objects
    private int nextId = 0;

    // Events
    public event Action<TrackedObject> onNewOCRCandidate;
    public event Action<List<TrackedObject>> onTrackedObjectsUpdated;

    private void OnEnable()
    {
        if (detectionPostProcessor != null)
        {
            detectionPostProcessor.onProcessedDetections += UpdateDetections;
        }
        /*
        if (decisionManager != null)
        {
            decisionManager.onDecisionMade += UpdateOCRResult;
        }
        */
    }

    private void OnDisable()
    {
        if (detectionPostProcessor != null)
        {
            detectionPostProcessor.onProcessedDetections -= UpdateDetections;
        }
        /*
        if (decisionManager != null)
        {
            decisionManager.onDecisionMade -= UpdateOCRResult;
        }
        */
    }

    private void Update()
    {
        // Decrease time to live for all objects
        foreach (var obj in trackedObjects)
        {
            obj.timeToLive -= Time.deltaTime;
        }

        // Remove expired objects
        RemoveExpired();
    }

    /*
        Main method to update tracking with new detections.
        - For each new detection, tries to match it with an existing tracked object using IOU.
        - If a match is found, updates the existing object; if not, creates a new tracked object.
        - Resets the TTL for matched objects and sends new candidates for OCR analysis.
    */
    private void UpdateDetections(List<DetectionData> detections)
    {
        foreach (var detection in detections)
        {
            // Try to match with existing tracked object
            TrackedObject matchedObject = MatchOrCreate(detection);

            // Update the object
            matchedObject.lastDetection = detection;
            matchedObject.timeToLive = timeToLive; // Reset TTL

            // If this is a new object and hasn't been analyzed, send for OCR
            if (!matchedObject.isAnalyzed)
            {
                onNewOCRCandidate?.Invoke(matchedObject);
            }
        }

        // Notify subscribers that tracking has been updated
        onTrackedObjectsUpdated?.Invoke(trackedObjects);
    }

    /*
        Tries to match a new detection with existing tracked objects using IOU.
        If a good match is found, returns the matched object; otherwise, creates and returns a new tracked object.
    */
    private TrackedObject MatchOrCreate(DetectionData detection)
    {
        float bestIOU = 0f;
        TrackedObject bestMatch = null;

        // Find best matching existing object
        foreach (var obj in trackedObjects)
        {
            float iou = CalculateIOU(detection.bboxNormalized, obj.lastDetection.bboxNormalized);

            if (iou > bestIOU && iou > iouThreshold)
            {
                bestIOU = iou;
                bestMatch = obj;
            }
        }

        // If good match found, return it
        if (bestMatch != null)
        {
            return bestMatch;
        }

        // No match - create new tracked object
        TrackedObject newObj = new TrackedObject
        {
            id = nextId++,
            lastDetection = detection,
            timeToLive = timeToLive,
            isAnalyzed = true, // For testing we set it to true to trigger OCR immediately
            text = "TEST AD", // For testing we set it to a dummy value to trigger OCR immediately
            shouldBlock = true, // For testing we set it to true to trigger blocking immediately
        };

        trackedObjects.Add(newObj);

        Debug.Log($"Created new TrackedObject with ID: {newObj.id}");

        return newObj;
    }

    /*
        Calculates the Intersection over Union (IOU) between two rectangles.
        Returns a value between 0 and 1, where 0 means no overlap and 1 means perfect overlap.
    */
    private float CalculateIOU(Rect a, Rect b)
    {
        // Calculate intersection
        float xOverlap = Mathf.Max(0, Mathf.Min(a.xMax, b.xMax) - Mathf.Max(a.xMin, b.xMin));
        float yOverlap = Mathf.Max(0, Mathf.Min(a.yMax, b.yMax) - Mathf.Max(a.yMin, b.yMin));
        float intersectionArea = xOverlap * yOverlap;

        // Calculate union
        float aArea = a.width * a.height;
        float bArea = b.width * b.height;
        float unionArea = aArea + bArea - intersectionArea;

        // Avoid division by zero
        if (unionArea == 0)
        {
            return 0f;
        }

        return intersectionArea / unionArea;
    }

    /*
        Called by DecisionManager with OCR results and blocking decision.
        Updates the TrackedObject with text and shouldBlock flag.
    */
    public void UpdateOCRResult(TrackedObject obj, string text, bool shouldBlock)
    {
        // Find the object in our list
        TrackedObject tracked = trackedObjects.Find(t => t.id == obj.id);

        if (tracked != null)
        {
            tracked.text = text;
            tracked.shouldBlock = shouldBlock;
            tracked.isAnalyzed = true;

            Debug.Log(
                $"Updated OCR result for object {obj.id}: '{text}', shouldBlock={shouldBlock}"
            );

            // Notify subscribers
            onTrackedObjectsUpdated?.Invoke(trackedObjects);
        }
        else
        {
            Debug.LogWarning(
                $"Tried to update OCR result for object {obj.id} but it's not in tracked list"
            );
        }
    }

    /*
        Removes objects whose time to live has expired.
        Logs how many objects were removed and notifies subscribers if any were removed.
    */
    private void RemoveExpired()
    {
        int removedCount = trackedObjects.RemoveAll(obj => obj.timeToLive <= 0);

        if (removedCount > 0)
        {
            Debug.Log($"Removed {removedCount} expired objects");

            // Notify subscribers that tracking has been updated
            onTrackedObjectsUpdated?.Invoke(trackedObjects);
        }
    }

    /*
        Public method to get a copy of the current list of tracked objects.
        This can be used by other components (e.g., for visualization) without risking modification of the internal list.
    */
    public List<TrackedObject> GetTrackedObjects()
    {
        return new List<TrackedObject>(trackedObjects); // Return a copy
    }
}
