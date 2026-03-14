/*
    TextRecognitionInference

    PURPOSE:
    Runs OCR model inference on cropped image tensors.

    FEATURES:
    - Async GPU readback
    - Coroutine structure prevents overlapping inference
*/
using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;
public class TextRecognitionInference : MonoBehaviour
{
    // Will be loaded with ONNX model
   [SerializeField] private ModelAsset modelAsset;

    // Will be loaded with cropped texture
   [SerializeField] private CroppedImageToTensor croppedImageToTensor;

   // Reference to the newest incoming tensor (older queued tensors are discarded)
   private Tensor<float> latestTensor;

   private FrameData latestFrame;

    // Worker runs Inference
   private Worker worker;

   public event Action<Tensor<float>, FrameData> sendOCRTensor;


    private void Awake()
    {
        if (modelAsset == null)
        {
            return;
        }

        var ocrModel = ModelLoader.Load(modelAsset);

        // GPUCompute backend → runs model on GPU
        worker = new Worker(ocrModel, BackendType.GPUCompute);
    }

    private void OnEnable()
    {
        if (croppedImageToTensor != null)
        {
            croppedImageToTensor.sendTensor += onNewTensor;           
        }
    }

    private void OnDisable()
    {
        if (croppedImageToTensor != null)
        {
            croppedImageToTensor.sendTensor -= onNewTensor;           
        }
    }


    /*
        Called whenever a cropped tensor is ready. 
        Only the latest arriving tensor will be used for inference
        Therefore each old one needs to be disposed to prevent memory leaks
    */
    private void onNewTensor(Tensor<float> tensor, FrameData frame) 
    {
        // Dispose queued tensor if still stored
        latestTensor?.Dispose();

        // Make latest tensor point to incoming tensor
        latestTensor = tensor;

        latestFrame = frame;
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

        sendOCRTensor?.Invoke(outputTensor, frame);
    }


    private void OnDestroy()
    {
        latestTensor?.Dispose();
        worker?.Dispose();
    }




}
