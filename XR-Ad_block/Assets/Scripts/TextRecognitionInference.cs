/*
    TextRecognitionInference

    PURPOSE:
    Runs Paddle OCR recognition model inference on image tensors.

    PIPELINE:
    ... → Post processing → THIS (OCR recognition) → Decoder -> ...

    FEATURES:
    - Converts Texture → Tensor on GPU
    - Async GPU readback (non-blocking)
    - Processes only latest frame (no queue)
    - Safe tensor disposal (no memory leaks)
*/
using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class TextRecognitionInference : MonoBehaviour
{
    // ONNX OCR model (assigned in Inspector)
    [SerializeField]
    private ModelAsset modelAsset;

    // TEMP: using camera frames directly instead of cropped text regions
    [SerializeField]
    private CaptureCameraFrame captureCameraFrame;

    /*
    Exact tensor input settings for ONNX model is: [DynamicDimension.0,3,48,DynamicDimension.1]
    In NCHW format
    Therefore tTW can be changed, with multiples of 32 being recommmended
    Default Batch Number should always be 1
    */

    [SerializeField]
    private int tensorTargetHeight = 48;

    [SerializeField]
    private int tensorTargetWidth = 320;

    // Reference to the newest incoming tensor (older queued tensors are discarded)
    private Tensor<float> latestTensor;

    // Metadata for corresponding frame
    private FrameData latestFrame;

    // Sentis worker → runs model on GPU
    private Worker worker;

    // Reused GPU resources for conversion
    private RenderTexture renderTexture;

    private CommandBuffer commandBuffer;

    public event Action<Tensor<float>, FrameData> sendOCRTensor;

    private void Awake()
    {
        if (modelAsset == null)
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
            captureCameraFrame.newFrame += onNewTensor;
        }
    }

    private void OnDisable()
    {
        if (captureCameraFrame != null)
        {
            captureCameraFrame.newFrame -= onNewTensor;
        }
    }

    /*
        Called whenever a cropped tensor is ready.
        Only the latest arriving tensor will be used for inference
        Therefore each old one needs to be disposed to prevent memory leaks
    */
    private void onNewTensor(FrameData frame)
    {
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

        Debug.Log("cash");
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

    //Runs inference asynchronously.
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

        // Disposes tensor used by inference
        inputTensor.Dispose();

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            yield break;
        }

        Debug.Log("skiiiiicka");

        sendOCRTensor?.Invoke(outputTensor, frame);
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
