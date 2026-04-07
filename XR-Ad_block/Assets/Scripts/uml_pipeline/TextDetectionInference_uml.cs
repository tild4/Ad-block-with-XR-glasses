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
public class TextDetectionInference_uml : MonoBehaviour
{
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private YOLOPostProcessor_uml yoloPostProcessor;

    private bool isProcessing = false;

    private Worker worker;

    public event Action<DetectionsPerAd> findTextRegions;


    private void Awake()
    {
        if (modelAsset == null || yoloPostProcessor == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        var ocrModel = ModelLoader.Load(modelAsset);
        worker = new Worker(ocrModel, BackendType.CPU);
    }

    private void OnEnable()
    {
        if (yoloPostProcessor != null)
        {
            yoloPostProcessor.onProcessedDetections += HandleNewTrackedObject;
        }
    }

    private void OnDisable()
    {
        if (yoloPostProcessor != null)
        {
            yoloPostProcessor.onProcessedDetections -= HandleNewTrackedObject;
        }
    }


    private void HandleNewTrackedObject(TrackedObject advertisement)
    {
        if (advertisement == null || advertisement.lastDetection.RoiTensor == null)
        {
            return;
        }

        if(!isProcessing)
        {
            StartCoroutine(RunOCRDetection(advertisement));   
        }
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
        Rect bounds = advertisement.lastDetection.bboxNormalized;

        if (inputTensor == null || worker == null)
        {
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

        //inputTensor.Dispose();

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            yield break;
        }

        //advertisement.findTextTensor = outputTensor;

        DetectionsPerAd findDetections = new DetectionsPerAd(advertisement,outputTensor);

        findTextRegions?.Invoke(findDetections);
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
