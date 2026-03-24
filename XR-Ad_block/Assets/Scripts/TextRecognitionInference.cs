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
using System.Collections.Generic;   // TEMP REMOVE! used for debugging!

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

    // Loaded with the yml file
    [SerializeField]
    private TextAsset ymlFile;

    // Reference to the newest incoming tensor (older queued tensors are discarded)
    private Tensor<float> latestTensor;

    // Metadata for corresponding frame
    private FrameData latestFrame;

    // Sentis worker → runs model on GPU
    private Worker worker;

    // Reused GPU resources for conversion
    private RenderTexture renderTexture;

    private CommandBuffer commandBuffer;

    private TextDecoder textDecoder;

    public event Action<Tensor<float>, FrameData> sendOCRTensor;

    private void Awake()
    {
        // CAPTURE CAMERA FRAME IS TEMP!
        if (modelAsset == null || ymlFile == null || captureCameraFrame == null)
        {
            return;
        }

        var ocrModel = ModelLoader.Load(modelAsset);

        textDecoder = new TextDecoder(ymlFile);

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

        PipelineProfiler.Begin("OCR TextRecog");
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

        PipelineProfiler.End("OCR TextRecog");

        // Disposes tensor used by inference
        inputTensor.Dispose();

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            yield break;
        }

        /*----------DEBUG----------*/
        Debug.Log($"OCR Output Shape: {outputTensor.shape}");

        for (int i = 0; i < Mathf.Min(5, outputTensor.shape[1]); i++)
        {
            string row = "";
            for (int j = 0; j < Mathf.Min(10, outputTensor.shape[2]); j++)
            {
                row += outputTensor[0, i, j].ToString("F2") + " ";
            }
            Debug.Log($"Timestep {i}: {row}");
        }

        Debug.Log($"Dict size: {textDecoder.DictionarySize}");
        Debug.Log($"Num classes: {outputTensor.shape[2]}");


        HashSet<int> seenIndices = new HashSet<int>();

        int numClasses = outputTensor.shape[2];
        int sequenceLength = outputTensor.shape[1];

        for (int i = 0; i < sequenceLength; i++)
        {
            float maxConfidence = float.MinValue;
            int maxIndex = -1;

            for (int j = 0; j < numClasses; j++)
            {
                float confidence = outputTensor[0, i, j];
                if (confidence > maxConfidence)
                {
                    maxConfidence = confidence;
                    maxIndex = j;
                }
            }

            seenIndices.Add(maxIndex);
            Debug.Log($"step {i}: maxIndex={maxIndex}, maxConfidence={maxConfidence:F4}");
        }

        Debug.Log("Predicted indices: " + string.Join(", ", seenIndices));
        /*----------DEBUG----------*/
        

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
