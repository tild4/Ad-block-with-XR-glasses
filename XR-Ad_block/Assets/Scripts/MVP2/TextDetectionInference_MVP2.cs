/*
    TextDetectionInference

    PURPOSE:
    Runs OCR text detection on the latest ROI tensor batch from YOLOPostProcessor.

    CURRENT FLOW:
    YOLOPostProcessor -> THIS -> ProcessOCRDetection2

    POLICY:
    - Latest batch wins.
    - Older pending batches are disposed before they start.
    - Inference runs sequentially per ROI.

    NOTE:
    Emitted tensor is a "heat map" of where text bounds might be relative to the cropped ad
*/

using System;
using System.Collections;
using System.Collections.Generic;
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

        var ocrModel = ModelLoader.Load(modelAsset);
        worker = new Worker(ocrModel, BackendType.CPU);
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
        FIX: Previously this method returned silently when the tensor was null,
        which meant findTextRegions was never fired. OCRPipelineManager listens
        to findTextRegions to reset its isProcessing flag — so a silent return
        here caused a permanent deadlock (the queue kept growing but nothing
        was ever dequeued again).

        Now we always fire findTextRegions with an empty DetectionsPerAd so the
        pipeline manager can move on to the next queued item.
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

        Tensor<float> normalizedInput = null;
        Tensor<float> outputTensor = null;
        bool inferenceSucceeded = false;

        // PP-OCR expects BGR channel order + ImageNet normalization.
        // Unity's tensor conversion produces RGB values in [0,1].
        // FIX: Wrapped in try/catch — if the input tensor was disposed by
        // TrackingManager between the null-check above and this point
        // (race with TTL expiry), ReadbackAndClone inside NormalizeForPPOCR
        // would throw. We catch it and signal completion to keep the pipeline alive.
        try
        {
            PipelineProfiler.begin("OCR Preprocess");
            normalizedInput = NormalizeForPPOCR(inputTensor);
            PipelineProfiler.end("OCR Preprocess");
        }
        catch (Exception e)
        {
            Debug.LogWarning(
                $"[TextDetect] Preprocessing failed for Object {advertisement.id}: {e.Message}"
            );
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

        PipelineProfiler.begin("OCR TextDetect");
        worker.Schedule(normalizedInput);

        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>)
            .ReadbackAndCloneAsync()
            .GetAwaiter();

        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        PipelineProfiler.end("OCR TextDetect");
        normalizedInput.Dispose();

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

    /*
        Normalizes an RGB [0,1] tensor for PP-OCR text detection.
        PP-OCR expects BGR channel order + ImageNet normalization:
        output = (pixel - mean) / std
    */
    private Tensor<float> NormalizeForPPOCR(Tensor<float> rgbTensor)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();

        Tensor<float> cpuTensor = rgbTensor.ReadbackAndClone();

        int height = cpuTensor.shape[2];
        int width = cpuTensor.shape[3];

        const float meanB = 0.485f;
        const float meanG = 0.456f;
        const float meanR = 0.406f;
        const float stdB = 0.229f;
        const float stdG = 0.224f;
        const float stdR = 0.225f;

        Tensor<float> result = new Tensor<float>(new TensorShape(1, 3, height, width));

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float r = cpuTensor[0, 0, y, x];
                float g = cpuTensor[0, 1, y, x];
                float b = cpuTensor[0, 2, y, x];

                result[0, 0, y, x] = (b - meanB) / stdB;
                result[0, 1, y, x] = (g - meanG) / stdG;
                result[0, 2, y, x] = (r - meanR) / stdR;
            }
        }

        cpuTensor.Dispose();

        Debug.Log(
            $"[OCR Preprocess] Normalized {height}x{width} RGB->BGR tensor in {sw.ElapsedMilliseconds}ms"
        );
        return result;
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
