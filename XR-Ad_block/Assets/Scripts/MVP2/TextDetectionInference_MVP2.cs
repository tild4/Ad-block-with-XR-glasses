/*
    TextDetectionInference

    PURPOSE:
    Runs OCR text detection on ROI tensors delivered by the OCR pipeline manager.

    CURRENT FLOW:
    OCRPipelineManager_MVP2 -> THIS -> ProcessOCRDetection_MVP2

    POLICY:
    - Starts one OCR text-detection inference at a time.
    - Always emits a completion event, even on early exits or failures.
    - Owns the ROI tensor/snapshot after detaching them from the tracked object's latest detection.

    NOTE:
    Emitted tensor is a "heat map" of where text bounds might be relative to the cropped ad
*/

using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;

public class TextDetectionInference_MVP2 : MonoBehaviour
{
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private OCRPipelineManager_MVP2 ocrPipelineManager;

    [SerializeField]
    private TrackingManager_MVP2 trackingManager;

    private bool isProcessing = false;

    private Worker worker;

    public event Action<DetectionsPerAd> findTextRegions;

    public event Action<TrackedObject, string, bool> onEarlyExitRequired; // Notify exit early if no text

    private void Awake()
    {
        if (modelAsset == null || ocrPipelineManager == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        var ocrModel = BuildPPOCRPreprocessedModel(ModelLoader.Load(modelAsset));
        worker = new Worker(ocrModel, BackendType.CPU);
    }

    private Model BuildPPOCRPreprocessedModel(Model sourceModel)
    {
        var graph = new FunctionalGraph();
        FunctionalTensor bgrInput = graph.AddInput(sourceModel, 0);

        FunctionalTensor mean = Functional.Constant(
            new TensorShape(1, 3, 1, 1),
            new[] { 0.485f, 0.456f, 0.406f }
        );
        FunctionalTensor std = Functional.Constant(
            new TensorShape(1, 3, 1, 1),
            new[] { 0.229f, 0.224f, 0.225f }
        );

        FunctionalTensor normalizedInput = (bgrInput - mean) / std;
        FunctionalTensor[] outputs = Functional.Forward(sourceModel, normalizedInput);
        graph.AddOutputs(outputs);

        return graph.Compile();
    }

    private void OnEnable()
    {
        if (ocrPipelineManager != null)
        {
            ocrPipelineManager.onReadyForOCR += HandleNewTrackedObject;
        }
    }

    private void OnDisable()
    {
        if (ocrPipelineManager != null)
        {
            ocrPipelineManager.onReadyForOCR -= HandleNewTrackedObject;
        }
    }

    // Allows external managers to ensure this component is subscribed to an OCR pipeline manager.
    // This makes wiring resilient if inspector fields weren't set in the scene.
    public void EnsureSubscribedTo(OCRPipelineManager_MVP2 mgr)
    {
        if (mgr == null)
            return;

        if (ocrPipelineManager == mgr)
            return;

        if (ocrPipelineManager != null)
        {
            ocrPipelineManager.onReadyForOCR -= HandleNewTrackedObject;
        }

        ocrPipelineManager = mgr;
        ocrPipelineManager.onReadyForOCR += HandleNewTrackedObject;
    }

    public void UnregisterFrom(OCRPipelineManager_MVP2 mgr)
    {
        if (mgr == null)
            return;

        if (ocrPipelineManager == mgr)
        {
            ocrPipelineManager.onReadyForOCR -= HandleNewTrackedObject;
            ocrPipelineManager = null;
        }
    }

    /*
        Handles one tracked object that is ready for OCR text detection.
        If the ROI tensor is missing, it still emits an empty result so the
        OCR pipeline manager can clear its busy state and continue.
    */
    private void HandleNewTrackedObject(TrackedObject advertisement)
    {
        if (advertisement == null || advertisement.lastDetection.RoiTensor == null)
        {
            // Signal completion with an empty result so that
            // OCRPipelineManager.OnOcrFinished resets isProcessing.
            Debug.LogWarning($"[TextDetect] Skipping null/disposed object, signalling completion.");
            findTextRegions?.Invoke(
                new DetectionsPerAd(advertisement, null, null, Rect.zero, Rect.zero)
            );
            return;
        }
        StartCoroutine(RunOCRDetection(advertisement));
    }

    private IEnumerator RunOCRDetection(TrackedObject advertisement)
    {
        // Prevents nested coroutines
        isProcessing = true;

        yield return RunInference(advertisement);

        isProcessing = false;
    }

    private IEnumerator RunInference(TrackedObject advertisement)
    {
        Tensor<float> inputTensor = null;
        RenderTexture roiSnapshot = null;
        Rect roiContentRect = Rect.zero;
        Rect yoloBounds = Rect.zero;

        // Capture values immediately before any yield.
        // TrackedObject.lastDetection is updated every YOLO frame (it's a shared
        // reference), so we freeze snapshot/bounds/tensor NOW while they still
        // correspond to each other. After a yield, lastDetection may already
        // point to a newer frame's data.
        //
        // FIX: Wrapped in try/catch because TrackedObject may have expired
        // (TTL) between being dequeued and reaching this point. In that case
        // its tensor could be disposed, causing an ObjectDisposedException.
        // Without this catch the coroutine would die silently and
        // findTextRegions would never fire, deadlocking the pipeline.
        try
        {
            inputTensor = advertisement.lastDetection.RoiTensor;
            roiContentRect = advertisement.lastDetection.RoiContentRectNormalized;
            roiSnapshot = advertisement.lastDetection.RoiSnapshot;
            yoloBounds = advertisement.lastDetection.bboxNormalized;
            // Transfer tensor/snapshot ownership to this coroutine so tracking can
            // safely update the object while OCR is running.
            advertisement.lastDetection.RoiTensor = null;
            // Setting RoiSnapshot to null transfers ownership to the OCR pipeline.
            advertisement.lastDetection.RoiSnapshot = null;
        }
        catch (Exception e)
        {
            Debug.LogWarning(
                $"[TextDetect] Failed to capture detection data for Object {advertisement.id}: {e.Message}"
            );
        }

        if (inputTensor == null || worker == null)
        {
            inputTensor?.Dispose();
            if (roiSnapshot != null)
            {
                roiSnapshot.Release();
                Destroy(roiSnapshot);
            }
            // Always signal completion so OCRPipelineManager.OnOcrFinished
            // resets isProcessing. Without this the pipeline permanently deadlocks.
            findTextRegions?.Invoke(
                new DetectionsPerAd(advertisement, null, null, Rect.zero, Rect.zero)
            );
            yield break;
        }

        Tensor<float> outputTensor = null;
        bool inferenceSucceeded = false;

        // The hot-path PPOCR preprocessing is now split between:
        // - ConvertToTensor.BgrChannelTransform in YOLOPostProcessor_MVP2
        // - the FunctionalGraph wrapper created in Awake for (pixel - mean) / std
        PipelineProfiler.begin("OCR Preprocess");
        PipelineProfiler.end("OCR Preprocess");

        Tensor<float> scheduledOutput = null;
        Awaitable<Tensor<float>> outputReadback;
        try
        {
            PipelineProfiler.begin("OCR TextDetect");
            worker.Schedule(inputTensor);
            scheduledOutput = worker.PeekOutput(0) as Tensor<float>;

            if (scheduledOutput == null)
            {
                throw new InvalidOperationException("OCR text detection output tensor was null.");
            }

            outputReadback = scheduledOutput.ReadbackAndCloneAsync();
        }
        catch (Exception e)
        {
            PipelineProfiler.end("OCR TextDetect");
            Debug.LogWarning(
                $"[TextDetect] Inference scheduling failed for Object {advertisement.id}: {e.Message}"
            );
            inputTensor.Dispose();
            if (roiSnapshot != null)
            {
                roiSnapshot.Release();
                Destroy(roiSnapshot);
            }
            findTextRegions?.Invoke(
                new DetectionsPerAd(advertisement, null, null, Rect.zero, Rect.zero)
            );
            yield break;
        }

        var outputAwaiter = outputReadback.GetAwaiter();

        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        PipelineProfiler.end("OCR TextDetect");
        inputTensor.Dispose();

        // FIX: Wrapped GetResult in try/catch — the async readback can fail
        // if the GPU resource was released during the wait (e.g. scene unload).
        // On failure we signal completion to prevent pipeline deadlock.
        try
        {
            outputTensor = outputAwaiter.GetResult();
            inferenceSucceeded = outputTensor != null;
        }
        catch (Exception e)
        {
            Debug.LogWarning(
                $"[TextDetect] Inference readback failed for Object {advertisement.id}: {e.Message}"
            );
        }

        if (inferenceSucceeded)
        {
            // check if heatmap contains any value above threshold to determine if text is likely present
            bool hasText = false;
            float[] tensorData = outputTensor.DownloadToArray();

            for (int i = 0; i < tensorData.Length; i++)
            {
                if (tensorData[i] > 0.2f)
                {
                    hasText = true;
                    break;
                }
            }

            if (!hasText)
            {
                Debug.Log(
                    $"[Early Exit] Early Exit for ID {advertisement.id} - no text found in heatmap."
                );

                // Dispose the output tensor immediately since we're not sending it to OCR, to free up resources.
                outputTensor.Dispose();
                if (roiSnapshot != null)
                {
                    roiSnapshot.Release();
                    Destroy(roiSnapshot);
                }

                // Notify tracking manager
                onEarlyExitRequired?.Invoke(advertisement, "", true);

                // Signal with an empty result to reset isProcessing and move on to the next item
                findTextRegions?.Invoke(
                    new DetectionsPerAd(advertisement, null, null, Rect.zero, Rect.zero)
                );
                yield break;
            }
        }
        else // On inference failed completely
        {
            if (roiSnapshot != null)
            {
                roiSnapshot.Release();
                Destroy(roiSnapshot);
            }
            findTextRegions?.Invoke(
                new DetectionsPerAd(advertisement, null, null, Rect.zero, Rect.zero)
            );
            yield break;
        }

        DetectionsPerAd findDetections = new DetectionsPerAd(
            advertisement,
            outputTensor,
            roiSnapshot,
            yoloBounds,
            roiContentRect
        );

        findTextRegions?.Invoke(findDetections);
        Debug.Log(
            $"[TextDetect] Heatmap generated for Object {advertisement.id}. Sending to ProcessOCRDetection."
        );
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
