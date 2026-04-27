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

    /*
        FIX: Changed from simple dequeue-one to a loop that skips stale entries.

        Previously this method dequeued exactly one item and sent it to OCR.
        If that item's tensor had been disposed (e.g. the TrackedObject expired
        via TTL while waiting in the queue), TextDetectionInference would
        silently return without firing findTextRegions. That left isProcessing
        stuck at true forever — a permanent deadlock where no further items
        could ever be processed.

        Now we loop through the queue, skipping any entry that is:
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
            var next = ocrQueue.Dequeue();

            // An object may have completed OCR via an earlier queue entry
            // (duplicates were possible before the TrackingManager fix).
            // Skip it to avoid redundant work.
            if (next == null || next.isAnalyzed)
            {
                Debug.Log($"[Queue] Skipping already-analyzed or null object. Remaining: {ocrQueue.Count}");
                continue;
            }

            // When a TrackedObject's TTL expires, RemoveExpired() disposes its
            // tensor but the queue still holds a reference to the object.
            // Attempting OCR on a disposed tensor would crash the inference
            // coroutine and deadlock the pipeline. Skip these safely.
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
                Debug.LogWarning($"[Queue] Skipping Object {next.id}: RoiTensor access failed (disposed?).");
                continue;
            }

            isProcessing = true;
            Debug.Log($"[Queue] Starting OCR process for Object {next.id}. Remaining: {ocrQueue.Count}");
            onReadyForOCR?.Invoke(next);
            return;
        }
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
