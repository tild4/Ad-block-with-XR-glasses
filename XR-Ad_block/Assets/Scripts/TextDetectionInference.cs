/*
    TextDetectionInference

    PURPOSE:
    Runs Paddle OCR detection model inference
    on ads detected from YOLO .

    PIPELINE POSITION:
    ... -> YOLO -> THIS (OCR detection) → Post Processing → OCR recognition -> ...

    FEATURES:
    - Converts camera Texture → Tensor on GPU
    - Throttled inference (processingInterval)
    - Only latest frame is processed (no queue)
    - Async GPU readback (non-blocking)
    - Safe tensor ownership & disposal
*/
using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class TextDetectionInference : MonoBehaviour
{
    // Will be loaded with ONNX model
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private CaptureCameraFrame captureCameraFrame;

    /*
    Exact tensor input settings for ONNX model is: [DynamicDimension.0,3,DynamicDimension.1,DynamicDimension.2]
    In NCHW format
    Therefore tTH and tTW can be changed, with multiples of 32 being recommmended
    Default Batch Number should always be 1
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

    // Reference to the newest incoming tensor (older queued tensors are discarded)
    private Tensor<float> latestTensor;

    // Metadata for corresponding frame
    private FrameData latestFrame;

    // Sentis worker → runs model on GPU
    private Worker worker;

    // Reused GPU resources for conversion
    private RenderTexture renderTexture;

    private CommandBuffer commandBuffer;

    // Output event: Sends detection output tensor + frame metadata
    public event Action<Tensor<float>, FrameData> decodeDetectionTensor;

    private void Awake()
    {
        // CAPTURE CAMERA FRAME IS TEMP!
        if (modelAsset == null || captureCameraFrame == null)
        {
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
        if (captureCameraFrame != null)
        {
            captureCameraFrame.newFrame += onNewFrame;
        }
    }

    private void OnDisable()
    {
        if (captureCameraFrame != null)
        {
            captureCameraFrame.newFrame -= onNewFrame;
        }
    }

    /*
        Called whenever a cropped tensor is ready.
        Only the latest arriving tensor will be used for inference
        Therefore each old one needs to be disposed to prevent memory leaks
    */
    private void onNewFrame(FrameData frame)
    {
        if (Time.time - lastProcessTime < processingInterval)
        {
            return;
        }

        lastProcessTime = Time.time;

        // Dispose previous tensor if it was never used. Prevents memory leaks.
        latestTensor?.Dispose();

        /*
            Convert current frame → tensor (GPU)
            Ownership is transferred to this class
        */
        latestTensor = ConvertToTensor.convert(
            frame.currentTexture,
            renderTexture,
            tensorTargetHeight,
            tensorTargetWidth,
            commandBuffer
        );

        latestFrame = frame;

        Debug.Log("yes min broder");
    }

    /*
        Coroutine that continuously runs inference attempts.
        runInference() internally decides whether work exists.
    */
    private IEnumerator Start()
    {
        while (true)
        {
            yield return runInference();
        }
    }

    /*
        Runs inference if a tensor is available.

        FLOW:
        1. Grab latest tensor
        2. Run GPU inference
        3. Await async readback
        4. Dispose input tensor
        5. Emit output tensor
    */
    private IEnumerator runInference()
    {
        if (latestTensor == null || worker == null)
        {
            yield return null;
            yield break;
        }

        FrameData frame = latestFrame;

        /*
         Transfer ownership safely of latest tensor to input tensor
         Input tensor points to the latest tensor
        */

        Tensor<float> inputTensor = latestTensor;

        // Make latest tensor point to null
        latestTensor = null;

        PipelineProfiler.Begin("OCR TextDetect");
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

        PipelineProfiler.End("OCR TextDetect");

        // Disposes tensor used by inference
        inputTensor.Dispose();

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            yield break;
        }

        Debug.Log("send to decode");

        decodeDetectionTensor?.Invoke(outputTensor, frame);
    }

    // Mandatory cleanup
    private void OnDestroy()
    {
        latestTensor?.Dispose();
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
