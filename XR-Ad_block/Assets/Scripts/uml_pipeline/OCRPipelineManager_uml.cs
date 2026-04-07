using System;
using System.Collections.Generic;
using UnityEngine;

public class OCRPipelineManager_uml : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private TrackingManager_uml trackingManager;

    [SerializeField]
    private TextDetectionInference_uml textDetectionInference;

    private Queue<TrackedObject> ocrQueue = new Queue<TrackedObject>();
    private bool isProcessing = false;

    // Emitted when an item is ready to be processed by OCR
    public event Action<TrackedObject> onReadyForOCR;

    private void OnEnable()
    {
        if (trackingManager != null)
        {
            trackingManager.onNewOCRCandidate += HandleNewCandidate;
        }

        if (textDetectionInference != null)
        {
            // Listen for OCR completion so we can continue the queue
            textDetectionInference.findTextRegions += OnOcrFinished;
            // Ensure TextDetectionInference is subscribed to this pipeline manager
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
            // Unregister wiring established at runtime
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
        if (ocrQueue.Count == 0)
            return;

        var next = ocrQueue.Dequeue();
        isProcessing = true;
        Debug.Log($"[Queue] Starting OCR process for Object {next.id}.");
        onReadyForOCR?.Invoke(next);
    }

    // Called when TextDetectionInference finishes processing an item
    private void OnOcrFinished(DetectionsPerAd result)
    {
        // result.trackedObject corresponds to the one we sent
        isProcessing = false;
        // Continue with next queued item
        ProcessNextInQueue();
    }
}
