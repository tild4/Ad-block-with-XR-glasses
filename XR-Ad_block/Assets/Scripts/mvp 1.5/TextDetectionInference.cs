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

public class TextDetectionInference : MonoBehaviour
{
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private YOLOPostProcessor yoloPostProcessor;

    [SerializeField]
    private float processingInterval = 0.3f;

    private float lastProcessTime = 0f;
    private bool isProcessing = false;

    private Worker worker;

    // New incoming work waits here until the queue coroutine picks it up.
    private List<YoloRoiTensor> pendingBatch;

    // Each entry is 1 ad including "heat map" tensor
    private readonly List<YoloRoiTensor> detectionTensorBatch = new List<YoloRoiTensor>();

    public event Action<List<YoloRoiTensor>> decodeDetectionTensors;

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
            yoloPostProcessor.onProcessedDetections += HandleNewDetectionBatch;
        }
    }

    private void OnDisable()
    {
        if (yoloPostProcessor != null)
        {
            yoloPostProcessor.onProcessedDetections -= HandleNewDetectionBatch;
        }
    }

    /*
        Receives a new ROI tensor batch from YOLO post-processing.

        FLOW:
        1. Filter invalid tensors.
        2. Keep only the newest pending batch.
        3. Start the processing coroutine if needed.
    */
    private void HandleNewDetectionBatch(List<YoloRoiTensor> detectionBatch)
    {
        if (detectionBatch == null || detectionBatch.Count == 0)
        {
            return;
        }

        List<YoloRoiTensor> filteredBatch = FilterValidDetections(detectionBatch);

        if (filteredBatch.Count == 0)
        {
            return;
        }

        ReplacePendingBatch(filteredBatch);

        if (!isProcessing)
        {
            StartCoroutine(ProcessQueue());
        }
    }

    /*
    For each YoloRoiTensor in the recieved batch:
    1. Iterate through tensors to check if null
    2. Add non-null tensors to new batch
    */
    private List<YoloRoiTensor> FilterValidDetections(List<YoloRoiTensor> detectionBatch)
    {
        List<YoloRoiTensor> batch = new List<YoloRoiTensor>();

        foreach (var item in detectionBatch)
        {
            if (item.Tensor != null)
            {
                batch.Add(item);
            }
        }

        return batch;
    }

    /*
    If pending batch is not null -> it is old -> should be replaced
    Dispose tensors safely from the batch
    Replace pending batch with this one
    */
    private void ReplacePendingBatch(List<YoloRoiTensor> latestBatch)
    {
        if (pendingBatch != null)
        {
            DisposeTensorBatch(pendingBatch);
        }

        pendingBatch = latestBatch;
    }

    /*
        Processes pending batches sequentially.

        POLICY:
        - Current batch finishes.
        - Any newer incoming batch replaces the next pending one.
    */
    private IEnumerator ProcessQueue()
    {
        // Prevents nested coroutines
        isProcessing = true;

        // Loop is alive as long as there is work
        while (pendingBatch != null)
        {
            // Delay
            // NOTE : pending batch can still be replaced at this point
            yield return WaitForProcessingInterval();
            lastProcessTime = Time.time;

            // Save latest batch
            List<YoloRoiTensor> batch = pendingBatch;
            pendingBatch = null;

            Debug.Log("Nr of items in ocr batch : " + batch.Count);

            // Run inference on each tensor
            foreach (var item in batch)
            {
                if (item.Tensor != null)
                {
                    yield return RunInference(item);
                }
            }

            // Safe copy batch for emission
            var sendBatch = new List<YoloRoiTensor>(detectionTensorBatch);
            decodeDetectionTensors?.Invoke(sendBatch);
            detectionTensorBatch.Clear();
        }

        isProcessing = false;
    }

    private IEnumerator WaitForProcessingInterval()
    {
        float timeSinceLastProcess = Time.time - lastProcessTime;

        if (timeSinceLastProcess < processingInterval)
        {
            yield return new WaitForSeconds(processingInterval - timeSinceLastProcess);
        }
    }

    /*
        Runs OCR text-detection inference for one ROI tensor.

        FLOW:
        1. Schedule inference.
        2. Wait for async readback.
        3. Dispose input tensor.
        4. Store output tensor for batch emission.
    */
    private IEnumerator RunInference(YoloRoiTensor roiInput)
    {
        Tensor<float> inputTensor = roiInput.Tensor;
        FrameData frame = roiInput.Frame;
        Rect bounds = roiInput.Bounds;

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

        inputTensor.Dispose();

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            yield break;
        }

        // Add heat map tensor, frame and Yolo bounds to emitted batch
        detectionTensorBatch.Add(new YoloRoiTensor(outputTensor, frame, bounds));
    }

    private void DisposeTensorBatch(List<YoloRoiTensor> batch)
    {
        if (batch == null)
        {
            return;
        }

        foreach (var item in batch)
        {
            item.Tensor?.Dispose();
        }
    }

    private void OnDestroy()
    {
        worker?.Dispose();

        if (pendingBatch != null)
        {
            DisposeTensorBatch(pendingBatch);
            pendingBatch = null;
        }

        if (detectionTensorBatch != null)
        {
            DisposeTensorBatch(detectionTensorBatch);
            detectionTensorBatch.Clear();
        }
    }
}
