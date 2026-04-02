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
        Model input shape:
        [Batch, Channels, Height, Width] (NCHW)

        - Batch = 1 (per ROI)
        - Height/Width should be multiples of 32 (model requirement)
    */

    [SerializeField]
    private int tensorTargetHeight = 640;

    [SerializeField]
    private int tensorTargetWidth = 640;

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

    // Reused GPU resources for conversion
    private RenderTexture renderTexture;

    private CommandBuffer commandBuffer;


    /*
        Holds latest batch of ROIs from YOLO

        IMPORTANT:
        - Overwritten on new detections
        - Only latest batch is processed
    */

    private List <(Texture, FrameData)> pendingBatch;


    /*
        Accumulates output tensors for current batch
        Cleared after emitting results
    */

    private List<(Tensor<float>, FrameData)> roiBatch = new List<(Tensor<float>, FrameData)>();

    // Output event (batch of detection tensors)
    public event Action<List<(Tensor<float>, FrameData)>> decodeDetectionTensors;

    private void Awake()
    {
        if (modelAsset == null || yoloPostProcessor == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        var ocrModel = ModelLoader.Load(modelAsset);

        //Allocate reusable GPU resources

        renderTexture = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        renderTexture.Create();

        commandBuffer = new CommandBuffer();

        // GPUCompute backend → runs model on GPU
        worker = new Worker(ocrModel, BackendType.GPUCompute);
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
    private void onNewDetection(List <(Texture, FrameData)> detection)
    {

        if (detection == null || detection.Count == 0)
        {
            return;
        }
        
        // Create safe copy (avoid mutation from producer)
        List<(Texture texture, FrameData frame)> batch =  new List<(Texture texture, FrameData frame)>();

        foreach (var item in detection)
        {
            Texture roi = item.Item1;
            FrameData frame = item.Item2;

            // add if texture exists
            if (roi != null)
            {
                batch.Add((roi, frame));
            }
        }

        if (batch.Count == 0)
        {
            return;
        }

        // Overwrite previous batch ("latest wins")
        pendingBatch = batch;

        // Start coroutine if not already running
        if (!isProcessing)
        {
            StartCoroutine(ProcessQueue());
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
            List<(Texture texture, FrameData frame)> batch = pendingBatch;

            pendingBatch = null;

            // runs inference on each item in batch
            foreach (var item in batch)
            {
                Texture roi = item.texture;
                FrameData frame = item.frame;

                if (roi == null)
                {
                    Debug.Log("bror du har blivit nullad");
                    continue;
                }

                Tensor<float> inputTensor = ConvertToTensor.convert(
                    roi,
                    renderTexture,
                    tensorTargetHeight,
                    tensorTargetWidth,
                    commandBuffer
                );

                if (inputTensor != null)
                {
                    yield return runInference(inputTensor, frame);
                } else if (inputTensor == null)
                {
                    Debug.Log("failed tensor conversion");
                }
            }


            /*
                Send batch results
                Copy list to avoid mutation issues
            */
            var sendBatch = new List<(Tensor<float>, FrameData)>(roiBatch);
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
    private IEnumerator runInference(Tensor<float> inputTensor, FrameData frame)
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

        roiBatch.Add((outputTensor, frame));
    }

    // Mandatory cleanup
    private void OnDestroy()
    {
        worker?.Dispose();

        if (commandBuffer != null)
        {
            commandBuffer.Release();
            commandBuffer = null;
        }

        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
            renderTexture = null;
        }
    }
}
