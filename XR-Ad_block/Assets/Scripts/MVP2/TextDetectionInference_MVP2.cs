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

        PipelineProfiler.begin("OCR TextDetect");
        worker.Schedule(inputTensor);

        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>)
            .ReadbackAndCloneAsync()
            .GetAwaiter();

        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        PipelineProfiler.end("OCR TextDetect");

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
            advertisement, outputTensor, roiSnapshot, yoloBounds
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
