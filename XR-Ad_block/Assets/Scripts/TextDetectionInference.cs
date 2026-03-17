/*
    TextRecognitionInference

    PURPOSE:
    Runs OCR detection model inference on cropped image tensors.

    FEATURES:
    - Async GPU readback
    - Coroutine structure prevents overlapping inference
*/
using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;
public class TextDetectionInference : MonoBehaviour
{
    // Will be loaded with ONNX model
   [SerializeField] private ModelAsset modelAsset;

   [SerializeField] private CaptureCameraFrame captureCameraFrame;

   [SerializeField] private int tensorTargetHeight = 640;

   [SerializeField] private int tensorTargetWidth = 640;

    [SerializeField] private float processingInterval = 0.3f;

    private float lastProcessTime = 0f;

   // Reference to the newest incoming tensor (older queued tensors are discarded)
   private Tensor<float> latestTensor;

   private FrameData latestFrame;

    // Worker runs Inference
   private Worker worker;

   private RenderTexture renderTexture;

   private CommandBuffer commandBuffer;

   public event Action<Tensor<float>, FrameData> decodeDetectionTensor;


    private void Awake()
    {
        if (modelAsset == null)
        {
            return;
        }

        var ocrModel = ModelLoader.Load(modelAsset);

        renderTexture = new RenderTexture(tensorTargetWidth, tensorTargetHeight, 0, RenderTextureFormat.ARGB32);

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


        // Dispose queued tensor if still stored
        latestTensor?.Dispose();

        // Make latest tensor point to incoming tensor
        latestTensor = ConvertToTensor.convert(frame.currentTexture, renderTexture, tensorTargetHeight, tensorTargetWidth, commandBuffer);

        latestFrame = frame;

        Debug.Log("yes min broder");
    }

    /*
        Coroutine that continuously runs inference attempts.
        runInference() internally decides whether work exists.
    */
    private IEnumerator Start()
    {
        while(true)
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
        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>).ReadbackAndCloneAsync().GetAwaiter();

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

        Debug.Log("send to decode");

        decodeDetectionTensor?.Invoke(outputTensor, frame);
    }


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
