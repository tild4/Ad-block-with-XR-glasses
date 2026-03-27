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
using System.Collections.Generic;
using System.Linq;   // TEMP REMOVE! used for debugging!

public class TextRecognitionInference : MonoBehaviour
{
    // ONNX OCR model (assigned in Inspector)
    [SerializeField]
    private ModelAsset modelAsset;

    // TEMP: using camera frames directly instead of cropped text regions
    [SerializeField]
    private ProcessOCRDetection getROIText;

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

    [SerializeField] ViewCroppedImage viewCroppedImage;

    // Sentis worker → runs model on GPU
    private Worker worker;

    // Reused GPU resources for conversion
    private RenderTexture renderTexture;

    private CommandBuffer commandBuffer;

    private TextDecoder textDecoder;

    private bool isProcessing = false;

    private Queue<(Texture texture, FrameData frame)> pendingItems =  new Queue<(Texture texture, FrameData frame)>();

    //public event Action<Tensor<float>, FrameData> sendOCRTensor;

    private void Awake()
    {
        if (modelAsset == null || ymlFile == null || getROIText == null)
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
        if (getROIText != null)
        {
            getROIText.sendCroppedROIText += onNewROI;
        }
    }

    private void OnDisable()
    {
        if (getROIText != null)
        {
            getROIText.sendCroppedROIText -= onNewROI;
        }
    }

    /*

    */
    private void onNewROI(List<Texture2D> croppedROI, FrameData frame)
    {
        if (isProcessing)
        {
            // Drop whole incoming batch
            foreach (Texture2D roi in croppedROI)
            {
                if (roi != null)
                {
                    Destroy(roi);
                }
            }
            return;
        }

        if (croppedROI != null && croppedROI.Count > 0 && viewCroppedImage != null)
        {
            Debug.Log("Now showing image!");
            viewCroppedImage.Show(croppedROI[0]);
            Debug.Log("image should be seen");
        }

        foreach (Texture2D roi in croppedROI)
        {
            if (roi != null)
            {
                pendingItems.Enqueue((roi, frame));
            }
        }

        if (pendingItems.Count > 0)
        {
            StartCoroutine(ProcessQueue());
        }
    }

    private IEnumerator ProcessQueue()
    {
        isProcessing = true;

        Debug.Log("Number of ROI this batch : " + pendingItems.Count);

        while (pendingItems.Count > 0)
        {
            var item = pendingItems.Dequeue();
            Texture roi = item.texture;
            FrameData frame = item.frame;

            Tensor<float> tensor = ConvertToTensor.convert(
                roi,
                renderTexture,
                tensorTargetHeight,
                tensorTargetWidth,
                commandBuffer
            );

            if (tensor != null)
            {
                yield return runInference(tensor, frame);
                tensor.Dispose();
            }

            if (roi is Texture2D roi2D)
            {
                Destroy(roi2D);
            }
        }

        isProcessing = false;
    }

    //Runs inference asynchronously.
    private IEnumerator runInference(Tensor<float> tensor, FrameData frame)
    {
        if (tensor == null || worker == null)
        {
            yield return null;
            yield break;
        }

        /*
         Transfer ownership safely of latest tensor to input tensor
         Input tensor points to the latest tensor
        */

        PipelineProfiler.begin("OCR TextRecog");
        worker.Schedule(tensor);

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

        PipelineProfiler.end("OCR TextRecog");

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            yield break;
        }
        
        /* ----------DEBUG---------------------
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
        */ 

        PipelineProfiler.begin("Text decoder");

        String output = textDecoder.decode(outputTensor);

        PipelineProfiler.end("Text decoder");

        outputTensor.Dispose();
        Debug.Log("Detected word is: " + output);
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
