/*
    Summary:
    Tracks YOLO detections over time, associates new detections with active
    objects, and emits updates for OCR and block placement.

    Pipeline:
    YOLOPostProcessor -> TrackingManager -> OCRPipelineManager,
    DecisionManager, BlockPlacementManager

    Note:
    This project uses and adapts sample code provided through the Meta XR SDK.

    Copyright © Meta Platform Technologies, LLC and its affiliates.
    All rights reserved.
*/

using System;
using System.Collections.Generic;
using UnityEngine;

public class TrackingManager : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private YOLOPostProcessor yoloPostProcessor;

    [SerializeField]
    private DecisionManager decisionManager;

    [SerializeField]
    private TextDetectionInference textDetectionInference;

    [SerializeField]
    private float instantYOLOThreshold = 0.8f;

    [Header("Tracking Settings")]
    [SerializeField]
    private float timeToLive = 2.0f;

    [SerializeField]
    private float iouThreshold = 0.5f;

    [Header("Debug")]
    [SerializeField]
    private bool logAssociationMisses;

    [SerializeField]
    private int maxTrackedObjects = 5;

    private List<TrackedObject> trackedObjects = new List<TrackedObject>();
    private int nextId = 0;

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
        if (textDetectionInference != null)
        {
            textDetectionInference.onEarlyExitRequired += UpdateOCRResult;
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
        if (textDetectionInference != null)
        {
            textDetectionInference.onEarlyExitRequired -= UpdateOCRResult;
        }
    }

    private void Update()
    {
        foreach (var obj in trackedObjects)
        {
            obj.timeToLive -= Time.deltaTime;
        }

        RemoveExpired();
    }

    private void UpdateDetections(List<DetectionData> detections)
    {
        Debug.Log($"[Tracking] Received {detections.Count} new detections from YOLO.");

        bool suppressNewObjects = trackedObjects.Count >= maxTrackedObjects;
        if (suppressNewObjects)
        {
            Debug.LogWarning(
                $"Max tracked objects ({maxTrackedObjects}) reached - suppressing new objects"
            );
        }

        foreach (var detection in detections)
        {
            bool allowCreate = !suppressNewObjects && trackedObjects.Count < maxTrackedObjects;
            bool wasNewlyCreated;
            TrackedObject matchedObject = MatchOrCreate(
                detection,
                allowCreate,
                out wasNewlyCreated
            );

            if (matchedObject == null)
            {
                continue;
            }

            var prevTensor = matchedObject.lastDetection.RoiTensor;
            if (prevTensor != null && prevTensor != detection.RoiTensor)
            {
                try
                {
                    prevTensor.Dispose();
                }
                catch (Exception) { }
            }

            var prevSnapshot = matchedObject.lastDetection.RoiSnapshot;
            if (prevSnapshot != null && prevSnapshot != detection.RoiSnapshot)
            {
                prevSnapshot.Release();
                Destroy(prevSnapshot);
            }

            matchedObject.lastDetection = detection;
            matchedObject.timeToLive = timeToLive;

            if (matchedObject.isAnalyzed && matchedObject.lastDetection.RoiSnapshot != null)
            {
                matchedObject.lastDetection.RoiSnapshot.Release();
                Destroy(matchedObject.lastDetection.RoiSnapshot);
                matchedObject.lastDetection.RoiSnapshot = null;
            }

            if (wasNewlyCreated)
            {
                onNewOCRCandidate?.Invoke(matchedObject);
            }
        }

        onTrackedObjectsUpdated?.Invoke(trackedObjects);
    }

    private TrackedObject MatchOrCreate(
        DetectionData detection,
        bool allowCreate,
        out bool wasNewlyCreated
    )
    {
        wasNewlyCreated = false;
        float bestIouOverall = float.NegativeInfinity;
        TrackedObject bestCandidate = null;
        Camera cam = Camera.main;

        // YOLO bboxes are top-left-origin; Unity viewport is bottom-left-origin.
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

        TrackedObject newObj = new TrackedObject
        {
            id = nextId++,
            lastDetection = detection,
            timeToLive = timeToLive,
            isAnalyzed = false,
            text = string.Empty,
            shouldBlock = detection.confidence >= instantYOLOThreshold,
        };

        trackedObjects.Add(newObj);
        wasNewlyCreated = true;

        Debug.Log($"Created new TrackedObject with ID: {newObj.id}");

        return newObj;
    }

    private static Rect ToViewportRect(Rect yoloNormalizedRect)
    {
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
        Quaternion relativeRot = Quaternion.Inverse(newPose.rotation) * oldPose.rotation;

        Quaternion camRot = cam.transform.rotation;
        Quaternion invCamRot = Quaternion.Inverse(camRot);

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
            Ray ray = cam.ViewportPointToRay(new Vector3(corner.x, corner.y, 0f));

            Vector3 localDir = invCamRot * ray.direction;
            Vector3 newLocalDir = relativeRot * localDir;

            if (newLocalDir.z <= 0f)
            {
                continue;
            }

            Vector3 newWorldDir = camRot * newLocalDir;
            Vector3 worldPoint = cam.transform.position + newWorldDir * 10f;
            Vector3 vp = cam.WorldToViewportPoint(worldPoint);

            minX = Mathf.Min(minX, vp.x);
            minY = Mathf.Min(minY, vp.y);
            maxX = Mathf.Max(maxX, vp.x);
            maxY = Mathf.Max(maxY, vp.y);
            anyVisible = true;
        }

        if (!anyVisible)
        {
            return new Rect(0, 0, 0, 0);
        }

        return new Rect(minX, minY, maxX - minX, maxY - minY);
    }

    private float CalculateIOU(Rect a, Rect b)
    {
        float xOverlap = Mathf.Max(0, Mathf.Min(a.xMax, b.xMax) - Mathf.Max(a.xMin, b.xMin));
        float yOverlap = Mathf.Max(0, Mathf.Min(a.yMax, b.yMax) - Mathf.Max(a.yMin, b.yMin));
        float intersectionArea = xOverlap * yOverlap;

        float aArea = a.width * a.height;
        float bArea = b.width * b.height;
        float unionArea = aArea + bArea - intersectionArea;

        if (unionArea == 0)
        {
            return 0f;
        }

        return intersectionArea / unionArea;
    }

    public void UpdateOCRResult(TrackedObject obj, string text, bool shouldBlock)
    {
        TrackedObject tracked = trackedObjects.Find(t => t.id == obj.id);

        if (tracked != null)
        {
            tracked.text = text;
            tracked.shouldBlock = shouldBlock;
            tracked.isAnalyzed = true;

            Debug.Log(
                $"Updated OCR result for object {obj.id}: '{text}', shouldBlock={shouldBlock}"
            );

            Debug.Log(
                $"[Tracking] Firing onTrackedObjectsUpdated, subscribers: {onTrackedObjectsUpdated?.GetInvocationList()?.Length ?? 0}"
            );
            onTrackedObjectsUpdated?.Invoke(trackedObjects);
        }
        else
        {
            Debug.LogWarning(
                $"Tried to update OCR result for object {obj.id} but it's not in tracked list"
            );
        }
    }

    private void RemoveExpired()
    {
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

    public List<TrackedObject> GetTrackedObjects()
    {
        return new List<TrackedObject>(trackedObjects);
    }
}
