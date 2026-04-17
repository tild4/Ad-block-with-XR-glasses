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

    private bool isProcessing = false;

    private Worker worker;

    public event Action<DetectionsPerAd> findTextRegions;

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

    private void HandleNewTrackedObject(TrackedObject advertisement)
    {
        if (advertisement == null || advertisement.lastDetection.RoiTensor == null)
        {
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
        Tensor<float> inputTensor = advertisement.lastDetection.RoiTensor;

        /*
            Freeze the ROI snapshot and YOLO bounds before any yield.
            TrackedObject.lastDetection is updated every YOLO frame,
            so we capture these values now while they still match the RoiTensor.
            Setting RoiSnapshot to null transfers ownership to the OCR pipeline.
        */
        Rect roiContentRect = advertisement.lastDetection.RoiContentRectNormalized;
        RenderTexture roiSnapshot = advertisement.lastDetection.RoiSnapshot;
        Rect yoloBounds = advertisement.lastDetection.bboxNormalized;
        advertisement.lastDetection.RoiSnapshot = null;

        if (inputTensor == null || worker == null)
        {
            if (roiSnapshot != null)
            {
                roiSnapshot.Release();
                Destroy(roiSnapshot);
            }
            yield return null;
            yield break;
        }

        // PP-OCR expects BGR channel order + ImageNet normalization.
        // Unity's tensor conversion produces RGB values in [0,1].
        PipelineProfiler.begin("OCR Preprocess");
        Tensor<float> normalizedInput = NormalizeForPPOCR(inputTensor);
        PipelineProfiler.end("OCR Preprocess");

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

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            if (roiSnapshot != null)
            {
                roiSnapshot.Release();
                Destroy(roiSnapshot);
            }
            yield break;
        }

        DetectionsPerAd findDetections = new DetectionsPerAd(
            advertisement, outputTensor, roiSnapshot, yoloBounds, roiContentRect
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
