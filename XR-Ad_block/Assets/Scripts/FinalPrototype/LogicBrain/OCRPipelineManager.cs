/*
    Summary:
    Queues tracked objects that need OCR and emits one valid candidate at a
    time to keep text detection serialized.

    Pipeline:
    TrackingManager -> OCRPipelineManager -> TextDetectionInference
*/
using System;
using System.Collections.Generic;
using UnityEngine;

public class OCRPipelineManager : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private TrackingManager trackingManager;

    [SerializeField]
    private TextDetectionInference textDetectionInference;

    private Queue<TrackedObject> ocrQueue = new Queue<TrackedObject>();
    private bool isProcessing = false;

    public event Action<TrackedObject> onReadyForOCR;

    private void OnEnable()
    {
        if (trackingManager != null)
        {
            trackingManager.onNewOCRCandidate += HandleNewCandidate;
        }

        if (textDetectionInference != null)
        {
            textDetectionInference.findTextRegions += OnOcrFinished;
            textDetectionInference.EnsureSubscribedTo(this);
        }
    }

    private void OnDisable()
    {
        if (trackingManager != null)
        {
            trackingManager.onNewOCRCandidate -= HandleNewCandidate;
        }

        if (textDetectionInference != null)
        {
            textDetectionInference.findTextRegions -= OnOcrFinished;
            textDetectionInference.UnregisterFrom(this);
        }
    }

    private void HandleNewCandidate(TrackedObject obj)
    {
        if (obj == null)
            return;
        ocrQueue.Enqueue(obj);
        Debug.Log($"[Queue] Object {obj.id} added to OCR queue. Queue size: {ocrQueue.Count}");
        ProcessNextInQueue();
    }

    private void ProcessNextInQueue()
    {
        if (isProcessing)
        {
            Debug.LogWarning("[Queue] Still waiting for previous OCR to finish...");
            return;
        }

        while (ocrQueue.Count > 0)
        {
            var next = ocrQueue.Dequeue();

            if (next == null || next.isAnalyzed)
            {
                Debug.Log(
                    $"[Queue] Skipping already-analyzed or null object. Remaining: {ocrQueue.Count}"
                );
                continue;
            }

            try
            {
                var tensor = next.lastDetection.RoiTensor;
                if (tensor == null)
                {
                    Debug.LogWarning($"[Queue] Skipping Object {next.id}: RoiTensor is null.");
                    continue;
                }
            }
            catch (Exception)
            {
                Debug.LogWarning(
                    $"[Queue] Skipping Object {next.id}: RoiTensor access failed (disposed?)."
                );
                continue;
            }

            isProcessing = true;
            Debug.Log(
                $"[Queue] Starting OCR process for Object {next.id}. Remaining: {ocrQueue.Count}"
            );
            onReadyForOCR?.Invoke(next);
            return;
        }
    }

    private void OnOcrFinished(DetectionsPerAd result)
    {
        isProcessing = false;
        ProcessNextInQueue();
    }
}
