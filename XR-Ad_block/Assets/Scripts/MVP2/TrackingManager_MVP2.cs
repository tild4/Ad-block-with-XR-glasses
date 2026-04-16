// Copyright (c) [Your Name or Project]. All rights reserved.
//
// This file contains code inspired by samples from Meta Platforms, Inc.
// It is not a direct copy of Meta source; the implementation and
// copyright belong to this project unless otherwise stated.
//
/*
    TrackingManager

    PURPOSE:
    The "Brain" of the persistence layer. It ensures that a physical object in the real
    world maintains the same unique ID across multiple camera frames, even if the
    AI detection momentarily flickers or disappears.

    ARCHITECTURE:
        - Object Association: Matches new detections with existing 'TrackedObjects'
            using pose-compensated IOU. The old bbox is reprojected into the new frame's
            screen space using the relative camera rotation between frames, then standard
            2D IOU is calculated.
    - ID Management: Assigns a permanent 'nextId' to every new unique detection.
    - Lifecycle Control:
        1. TTL (Time To Live): Keeps objects alive for a few seconds (e.g., 2.0s)
           after they are no longer seen by the AI.
        2. Cleanup: Automatically purges expired objects from the system.
    - OCR Coordination: Identifies new objects that haven't been analyzed yet and
      broadcasts them via 'onNewOCRCandidate'.

    IMPORTANT:
        - The 'iouThreshold' controls matching sensitivity.
        - Reprojection uses only rotation (ignores translation parallax).
    - 'UpdateDetections' is the main entry point, triggered by the DetectionPostProcessor.
*/

using System;
using System.Collections.Generic;
using UnityEngine;

public class TrackingManager_MVP2 : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private YOLOPostProcessor_MVP2 yoloPostProcessor;

    [SerializeField]
    private DecisionManager decisionManager;

    [Header("Tracking Settings")]
    [SerializeField]
    private float timeToLive = 2.0f; // Seconds before object expires

    [SerializeField]
    private float iouThreshold = 0.5f; // Threshold for matching detections

    [Header("Debug")]
    [SerializeField]
    private bool logAssociationMisses;

    [SerializeField]
    private int maxTrackedObjects = 5; // Limit to prevent overload for testing

    // List of currently tracked objects
    private List<TrackedObject> trackedObjects = new List<TrackedObject>();

    // ID counter for new objects
    private int nextId = 0;

    // Events
    public event Action<TrackedObject> onNewOCRCandidate;
    public event Action<List<TrackedObject>> onTrackedObjectsUpdated;

    private void OnEnable()
    {
        if (yoloPostProcessor != null)
        {
            yoloPostProcessor.onProcessedDetections += UpdateDetections;
        }
        if (decisionManager != null)
        {
            decisionManager.onDecisionMade += UpdateOCRResult;
        }
    }

    private void OnDisable()
    {
        if (yoloPostProcessor != null)
        {
            yoloPostProcessor.onProcessedDetections -= UpdateDetections;
        }
        if (decisionManager != null)
        {
            decisionManager.onDecisionMade -= UpdateOCRResult;
        }
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
        Debug.Log($"[Tracking] Received {detections.Count} new detections from YOLO.");
        // If we're at capacity, keep matching/updating existing objects but don't create new ones.
        // Otherwise tracked objects would stop receiving TTL refreshes and expire, causing IDs to
        // continuously increase as objects get recreated.
        bool suppressNewObjects = trackedObjects.Count >= maxTrackedObjects;
        if (suppressNewObjects)
        {
            Debug.LogWarning(
                $"Max tracked objects ({maxTrackedObjects}) reached - suppressing new objects"
            );
        }

        foreach (var detection in detections)
        {
            // Try to match with existing tracked object
            bool allowCreate = !suppressNewObjects && trackedObjects.Count < maxTrackedObjects;
            TrackedObject matchedObject = MatchOrCreate(detection, allowCreate);

            if (matchedObject == null)
            {
                // At capacity and no match found - ignore this detection.
                continue;
            }

            // Dispose previous tensor if it's different from the incoming one
            var prevTensor = matchedObject.lastDetection.RoiTensor;
            if (prevTensor != null && prevTensor != detection.RoiTensor)
            {
                try
                {
                    prevTensor.Dispose();
                }
                catch (Exception) { }
            }

            // Release old ROI snapshot if still present (OCR pipeline didn't claim it)
            var prevSnapshot = matchedObject.lastDetection.RoiSnapshot;
            if (prevSnapshot != null && prevSnapshot != detection.RoiSnapshot)
            {
                prevSnapshot.Release();
                Destroy(prevSnapshot);
            }

            // Update the object
            matchedObject.lastDetection = detection;
            matchedObject.timeToLive = timeToLive; // Reset TTL

            // Analyzed objects don't need OCR; release snapshot immediately
            if (matchedObject.isAnalyzed && matchedObject.lastDetection.RoiSnapshot != null)
            {
                matchedObject.lastDetection.RoiSnapshot.Release();
                Destroy(matchedObject.lastDetection.RoiSnapshot);
                matchedObject.lastDetection.RoiSnapshot = null;
            }

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
    private TrackedObject MatchOrCreate(DetectionData detection, bool allowCreate)
    {
        float bestIouOverall = float.NegativeInfinity;
        TrackedObject bestCandidate = null;
        Camera cam = Camera.main;

        // YOLO bboxes are top-left-origin; Unity viewport is bottom-left-origin.
        // For any viewport-based reprojection/IOU, we must compare in the same coordinate system.
        Rect detectionViewportRect = ToViewportRect(detection.bboxNormalized);

        foreach (var obj in trackedObjects)
        {
            Rect trackedViewportRect = ToViewportRect(obj.lastDetection.bboxNormalized);

            float iou;

            if (cam != null)
            {
                Rect reprojected = ReprojectBbox(
                    trackedViewportRect,
                    obj.lastDetection.frame.currentPose,
                    detection.frame.currentPose,
                    cam
                );

                // Reprojection can produce an empty/invalid rect (e.g., corners behind camera).
                // In that case, fall back to a simple IOU in the current viewport space.
                if (reprojected.width <= 0f || reprojected.height <= 0f)
                {
                    iou = CalculateIOU(detectionViewportRect, trackedViewportRect);
                }
                else
                {
                    iou = CalculateIOU(detectionViewportRect, reprojected);
                }
            }
            else
            {
                // Fallback without camera: raw IOU in the YOLO coordinate system.
                iou = CalculateIOU(detection.bboxNormalized, obj.lastDetection.bboxNormalized);
            }

            if (iou > bestIouOverall)
            {
                bestIouOverall = iou;
                bestCandidate = obj;
            }
        }

        if (bestCandidate != null && bestIouOverall >= iouThreshold)
        {
            return bestCandidate;
        }

        if (logAssociationMisses)
        {
            string bestId = bestCandidate != null ? bestCandidate.id.ToString() : "none";
            Debug.Log(
                $"Association miss: bestIOU={bestIouOverall:0.000} < threshold={iouThreshold:0.000}, bestCandidateId={bestId}, allowCreate={allowCreate}"
            );
        }

        if (!allowCreate)
        {
            return null;
        }

        // No match - create new tracked object
        TrackedObject newObj = new TrackedObject
        {
            id = nextId++,
            lastDetection = detection,
            timeToLive = timeToLive,
            isAnalyzed = false,
            text = string.Empty,
            shouldBlock = detection.confidence >= 0.8f
        };

        trackedObjects.Add(newObj);

        Debug.Log($"Created new TrackedObject with ID: {newObj.id}");

        return newObj;
    }

    private static Rect ToViewportRect(Rect yoloNormalizedRect)
    {
        // YOLO: (x, y) = top-left. Viewport: (x, y) = bottom-left.
        // Keep width/height unchanged; only Y origin flips.
        float viewportYMin = 1f - yoloNormalizedRect.yMax;
        return new Rect(
            yoloNormalizedRect.xMin,
            viewportYMin,
            yoloNormalizedRect.width,
            yoloNormalizedRect.height
        );
    }

    private static Rect ReprojectBbox(Rect oldViewportRect, Pose oldPose, Pose newPose, Camera cam)
    {
        // Relative rotation from old frame to new frame
        Quaternion relativeRot = Quaternion.Inverse(newPose.rotation) * oldPose.rotation;

        // Current camera rotation (used as intermediate reference, cancels out)
        Quaternion camRot = cam.transform.rotation;
        Quaternion invCamRot = Quaternion.Inverse(camRot);

        // 4 corners of old bbox in viewport space
        Vector2 bottomLeft = new Vector2(oldViewportRect.xMin, oldViewportRect.yMin);
        Vector2 bottomRight = new Vector2(oldViewportRect.xMax, oldViewportRect.yMin);
        Vector2 topRight = new Vector2(oldViewportRect.xMax, oldViewportRect.yMax);
        Vector2 topLeft = new Vector2(oldViewportRect.xMin, oldViewportRect.yMax);

        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;

        Vector2[] corners = { bottomLeft, bottomRight, topRight, topLeft };
        bool anyVisible = false;

        foreach (var corner in corners)
        {
            // Viewport → world ray direction (depends on current cam rotation + FOV)
            Ray ray = cam.ViewportPointToRay(new Vector3(corner.x, corner.y, 0f));

            // World direction → camera-local direction (removes current cam rotation)
            Vector3 localDir = invCamRot * ray.direction;

            // Apply relative rotation (old frame → new frame)
            Vector3 newLocalDir = relativeRot * localDir;

            // Skip corners that end up behind the camera
            if (newLocalDir.z <= 0f)
            {
                continue;
            }

            // Camera-local → world direction (re-applies current cam rotation)
            Vector3 newWorldDir = camRot * newLocalDir;

            // Project back to viewport: use a point along the direction
            Vector3 worldPoint = cam.transform.position + newWorldDir * 10f;
            Vector3 vp = cam.WorldToViewportPoint(worldPoint);

            minX = Mathf.Min(minX, vp.x);
            minY = Mathf.Min(minY, vp.y);
            maxX = Mathf.Max(maxX, vp.x);
            maxY = Mathf.Max(maxY, vp.y);
            anyVisible = true;
        }

        // If no corners are visible, return empty rect (no match possible)
        if (!anyVisible)
        {
            return new Rect(0, 0, 0, 0);
        }

        return new Rect(minX, minY, maxX - minX, maxY - minY);
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
            Debug.Log($"[Tracking] Firing onTrackedObjectsUpdated, subscribers: {onTrackedObjectsUpdated?.GetInvocationList()?.Length ?? 0}"); // Pontus Debugging
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
        // Find expired objects first so we can dispose their tensors
        var expired = new List<TrackedObject>();
        foreach (var obj in trackedObjects)
        {
            if (obj.timeToLive <= 0)
            {
                expired.Add(obj);
            }
        }

        if (expired.Count == 0)
            return;

        foreach (var obj in expired)
        {
            try
            {
                obj.lastDetection.RoiTensor?.Dispose();
            }
            catch (Exception) { }

            if (obj.lastDetection.RoiSnapshot != null)
            {
                obj.lastDetection.RoiSnapshot.Release();
                Destroy(obj.lastDetection.RoiSnapshot);
            }
        }

        trackedObjects.RemoveAll(obj => obj.timeToLive <= 0);

        Debug.Log($"Removed {expired.Count} expired objects");
        onTrackedObjectsUpdated?.Invoke(trackedObjects);
    }

    private void OnDestroy()
    {
        // Dispose any tensors and snapshots still held by tracked objects
        foreach (var obj in trackedObjects)
        {
            try
            {
                obj.lastDetection.RoiTensor?.Dispose();
            }
            catch (Exception) { }

            if (obj.lastDetection.RoiSnapshot != null)
            {
                obj.lastDetection.RoiSnapshot.Release();
                Destroy(obj.lastDetection.RoiSnapshot);
            }
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
