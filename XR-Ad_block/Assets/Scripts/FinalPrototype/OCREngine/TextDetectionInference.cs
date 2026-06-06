/*
    Summary:
    Runs PP-OCR text detection on YOLO ROI tensors and emits heatmaps for
    text box extraction.

    Pipeline:
    OCRPipelineManager -> TextDetectionInference -> ProcessOCRDetection
*/

using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;

public class TextDetectionInference : MonoBehaviour
{
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private OCRPipelineManager ocrPipelineManager;

    private Worker worker;

    public event Action<DetectionsPerAd> findTextRegions;

    public event Action<TrackedObject, string, bool> onEarlyExitRequired;

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

    public void EnsureSubscribedTo(OCRPipelineManager mgr)
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

    public void UnregisterFrom(OCRPipelineManager mgr)
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
            Debug.LogWarning($"[TextDetect] Skipping null/disposed object, signalling completion.");
            findTextRegions?.Invoke(new DetectionsPerAd(advertisement, null, null));
            return;
        }
        StartCoroutine(RunOCRDetection(advertisement));
    }

    private IEnumerator RunOCRDetection(TrackedObject advertisement)
    {
        yield return RunInference(advertisement);
    }

    private IEnumerator RunInference(TrackedObject advertisement)
    {
        Tensor<float> inputTensor = null;
        RenderTexture roiSnapshot = null;

        // Capture the current tensor and snapshot before yielding; tracking can
        // update the same object while OCR is running.
        try
        {
            inputTensor = advertisement.lastDetection.RoiTensor;
            roiSnapshot = advertisement.lastDetection.RoiSnapshot;
            advertisement.lastDetection.RoiTensor = null;
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

            findTextRegions?.Invoke(new DetectionsPerAd(advertisement, null, null));
            yield break;
        }

        Tensor<float> outputTensor = null;
        bool inferenceSucceeded = false;

        // PPOCR normalization is wrapped into the model graph in Awake.
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
            findTextRegions?.Invoke(new DetectionsPerAd(advertisement, null, null));
            yield break;
        }

        var outputAwaiter = outputReadback.GetAwaiter();

        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        PipelineProfiler.end("OCR TextDetect");
        inputTensor.Dispose();

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

                outputTensor.Dispose();
                if (roiSnapshot != null)
                {
                    roiSnapshot.Release();
                    Destroy(roiSnapshot);
                }

                onEarlyExitRequired?.Invoke(advertisement, "", true);

                findTextRegions?.Invoke(new DetectionsPerAd(advertisement, null, null));
                yield break;
            }
        }
        else
        {
            if (roiSnapshot != null)
            {
                roiSnapshot.Release();
                Destroy(roiSnapshot);
            }
            findTextRegions?.Invoke(new DetectionsPerAd(advertisement, null, null));
            yield break;
        }

        DetectionsPerAd findDetections = new DetectionsPerAd(
            advertisement,
            outputTensor,
            roiSnapshot
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
