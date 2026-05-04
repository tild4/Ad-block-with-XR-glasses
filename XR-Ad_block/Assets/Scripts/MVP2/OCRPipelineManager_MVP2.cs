/*
    OCRPipelineManager

    This component manages the flow of objects from the TrackingManager to the TextDetectionInference.
    It maintains a queue of candidates for OCR and ensures that only one item is processed at a time.

    Responsibilities:
    - Listen for new OCR candidates from the TrackingManager and enqueue them.
    - Emit an event when an item is ready to be processed by OCR.
    - Listen for completion of OCR processing to continue with the next item in the queue.
*/
using System;
using System.Collections.Generic;
using UnityEngine;

public class OCRPipelineManager_MVP2 : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private TrackingManager_MVP2 trackingManager;

    [SerializeField]
    private TextDetectionInference_MVP2 textDetectionInference;

    // Queue to manage OCR candidates
    private Queue<TrackedObject> ocrQueue = new Queue<TrackedObject>();
    private bool isProcessing = false;

    // Emitted when an item is ready to be processed by OCR
    public event Action<TrackedObject> onReadyForOCR;

    /*
        Subscribes to TrackingManager and OCR inference events when enabled.
    */
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

    /*
        Unsubscribes from events when disabled to avoid leaks/duplicate subscriptions.
    */
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

    /*
        When TrackingManager identifies a new candidate for OCR, it invokes onNewOCRCandidate.
        We enqueue the candidate and attempt to process the next item in the queue.
    */
    private void HandleNewCandidate(TrackedObject obj)
    {
        if (obj == null)
            return;
        ocrQueue.Enqueue(obj);
        Debug.Log($"[Queue] Object {obj.id} added to OCR queue. Queue size: {ocrQueue.Count}");
        ProcessNextInQueue();
    }

    /*
        Loop through the queue, skipping any entry that is:
          - null or already analyzed (OCR result already obtained)
          - missing its RoiTensor (expired object whose tensor was disposed)
        This guarantees we either find a valid item to process or drain the
        queue cleanly.
    */
    private void ProcessNextInQueue()
    {
        if (isProcessing)
        {
            Debug.LogWarning("[Queue] Still waiting for previous OCR to finish...");
            return;
        }

        while (ocrQueue.Count > 0)
        {
            // Dequeue the next candidate for OCR processing
            var next = ocrQueue.Dequeue();

            // Skip any null or already-analyzed objects
            if (next == null || next.isAnalyzed)
            {
                Debug.Log(
                    $"[Queue] Skipping already-analyzed or null object. Remaining: {ocrQueue.Count}"
                );
                continue;
            }

            // Skip any object whose RoiTensor is inaccessible (disposed) or null (never set)
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
            // Found a valid candidate for OCR processing
            isProcessing = true;
            Debug.Log(
                $"[Queue] Starting OCR process for Object {next.id}. Remaining: {ocrQueue.Count}"
            );
            onReadyForOCR?.Invoke(next);
            return;
        }
    }

    /*
        When TextDetectionInference finishes processing an item, it invokes findTextRegions.
        We set isProcessing to false and attempt to process the next item in the queue.
    */
    private void OnOcrFinished(DetectionsPerAd result)
    {
        // result.trackedObject corresponds to the one we sent
        isProcessing = false;
        // Continue with next queued item
        ProcessNextInQueue();
    }
}
