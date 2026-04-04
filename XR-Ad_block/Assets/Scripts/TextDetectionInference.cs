/*
    TextDetectionInference

    PURPOSE:
    Runs OCR text detection model (e.g. PaddleOCR detection)
    on regions of interest (ROIs) provided by YOLO.

    PIPELINE:
    Camera → YOLO → ROI extraction → THIS → Post-processing → OCR recognition

    FEATURES:
    - Processes batches of cropped ROIs (not full frames)
    - Converts Texture → Tensor on GPU
    - Throttled processing (processingInterval)
    - "Latest batch wins" (older batches overwritten)
    - Sequential inference per ROI (no parallel GPU overload)
    - Async GPU readback (non-blocking)
    - Safe tensor ownership & disposal

    DESIGN:
    - pendingBatch stores latest incoming detections
    - Coroutine processes batch safely over time
    - roiBatch accumulates outputs before emitting
*/
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class TextDetectionInference : MonoBehaviour
{
    // Will be loaded with ONNX model
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private YOLOPostProcessor yoloPostProcessor;

    /*
        Minimum time between inference runs (seconds)
        Prevents running every frame
    */

    [SerializeField]
    private float processingInterval = 0.3f;

    private float lastProcessTime = 0f;

    private bool isProcessing = false;

    // Sentis worker → runs model on GPU
    private Worker worker;


    /*
        Holds latest batch of ROIs from YOLO

        IMPORTANT:
        - Overwritten on new detections
        - Only latest batch is processed
    */

    private List <(Tensor<float> roiTensor, FrameData, Rect)> pendingBatch;


    /*
        Accumulates output tensors for current batch
        Cleared after emitting results
    */

    private List<(Tensor<float>, FrameData, Rect)> roiBatch = new List<(Tensor<float>, FrameData, Rect)>();

    // Output event (batch of detection tensors)
    public event Action<List<(Tensor<float>, FrameData, Rect)>> decodeDetectionTensors;

    private void Awake()
    {
        if (modelAsset == null || yoloPostProcessor == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        var ocrModel = ModelLoader.Load(modelAsset);

        // CPU
        worker = new Worker(ocrModel, BackendType.CPU);
    }

    private void OnEnable()
    {
        if (yoloPostProcessor!= null)
        {
            yoloPostProcessor.onProcessedDetections += onNewDetection;
        }
    }

    private void OnDisable()
    {
        if (yoloPostProcessor != null)
        {
            yoloPostProcessor.onProcessedDetections -= onNewDetection;
        }
    }

    /*
        Called when YOLO produces a new batch of ROIs.

        RESPONSIBILITIES:
        - Filter invalid textures
        - Copy batch (avoid shared reference issues)
        - Store as latest batch
        - Start processing if not already running
    */
    private void onNewDetection(List <(Tensor<float>, FrameData, Rect)> detection)
    {

        if (detection == null || detection.Count == 0)
        {
            return;
        }
    
        Debug.Log("New roi from YOLO PP");
        
        // Create safe copy 
        List<(Tensor<float> roiTensor, FrameData frame, Rect yoloBounds)> batch =  new List<(Tensor<float> roiTensor, FrameData frame, Rect yoloBounds)>();

        PipelineProfiler.begin("Tensor filter 1");
        foreach (var item in detection)
        {
            Tensor<float> roiTensor = item.Item1;
            FrameData frame = item.Item2;
            Rect bounds = item.Item3;

            if (roiTensor != null)
            {
                batch.Add((roiTensor, frame, bounds));
            }
        }
        PipelineProfiler.end("Tensor filter 1");

        if (batch.Count == 0)
        {
            return;
        }

        PipelineProfiler.begin("Pending batch disposal");
        // If an older batch was waiting but never processed, drop it safely
        if (pendingBatch != null)
        {
            DisposeTensorBatch(pendingBatch);
        }
        PipelineProfiler.end("Pending batch disposal");

        // Overwrite previous batch ("latest wins")
        pendingBatch = batch;

        // Start coroutine if not already running
        if (!isProcessing)
        {
            PipelineProfiler.begin("Process queue det");
            StartCoroutine(ProcessQueue());
            PipelineProfiler.end("Process queue det");
        }
    }


    /*
        Processes batches sequentially.

        KEY IDEA:
        - Always processes latest batch
        - New incoming batches overwrite pendingBatch
    */
        private IEnumerator ProcessQueue()
    {
        isProcessing = true;

        // Throttle processing rate
        while (pendingBatch != null)
        {
            float timeSinceLastProcess = Time.time - lastProcessTime;

            if (timeSinceLastProcess < processingInterval)
            {
                // Yields control -> new batch can arrive in pending batches
                yield return new WaitForSeconds(processingInterval - timeSinceLastProcess);
            }

            lastProcessTime = Time.time;

            // Saves the latest batch
            List<(Tensor<float> roiTensor, FrameData frame, Rect yoloBounds)> batch = pendingBatch;

            pendingBatch = null;
            Debug.Log("Nr of items in ocr batch : " + batch.Count);

            // runs inference on each item in batch
            foreach (var item in batch)
            {
                Tensor<float> inputTensor = item.roiTensor;
                FrameData frame = item.frame;
                Rect bounds = item.yoloBounds;

                if (inputTensor != null)
                {
                    yield return runInference(inputTensor, frame, bounds
                    );
                } 
                else if (inputTensor == null)
                {
                    Debug.Log("failed tensor conversion");
                }
            }

            /*
                Send batch results
                Copy list to avoid mutation issues
            */
            var sendBatch = new List<(Tensor<float>, FrameData, Rect)>(roiBatch);
            decodeDetectionTensors?.Invoke(sendBatch);


            // Clear for next batch
            roiBatch.Clear();
        }
        isProcessing = false;
    }


    /*
        Runs inference for ONE ROI.

        FLOW:
        1. Schedule GPU inference
        2. Await async readback
        3. Dispose input tensor
        4. Store output
    */
    private IEnumerator runInference(Tensor<float> inputTensor, FrameData frame, Rect bound)
    {
        if (inputTensor == null || worker == null)
        {
            yield return null;
            yield break;
        }

        PipelineProfiler.begin("OCR TextDetect");
        worker.Schedule(inputTensor);

        /*
            Async GPU readback.
            Does NOT block main thread.
        */
        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>)
            .ReadbackAndCloneAsync()
            .GetAwaiter();

        // Loop until GPU has finished computing
        while (!outputAwaiter.IsCompleted)
        {
            // Pause execution, resume next FRAME
            yield return null;
        }

        PipelineProfiler.end("OCR TextDetect");

        // Disposes tensor used by inference
        inputTensor.Dispose();

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            yield break;
        }

        // Store result for batch emission

        roiBatch.Add((outputTensor, frame, bound));
    }

    private void DisposeTensorBatch(List<(Tensor<float>, FrameData, Rect)> batch)
    {
        if (batch == null)
        {
            return;
        }

        foreach (var item in batch)
        {
            item.Item1?.Dispose();
        }
    }

    // Mandatory cleanup
    private void OnDestroy()
    {
        worker?.Dispose();

        if (pendingBatch != null)
        {
            DisposeTensorBatch(pendingBatch);
            pendingBatch = null;
        }

        if (roiBatch != null)
        {
            DisposeTensorBatch(roiBatch);
            roiBatch.Clear();
        }
    }
}
