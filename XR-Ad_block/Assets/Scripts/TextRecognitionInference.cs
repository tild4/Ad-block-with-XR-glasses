/*
    TextRecognitionInference

    PURPOSE:
    Runs Paddle OCR recognition model inference on cropped text ROIs.

    PIPELINE:
    ... → Post processing → THIS (OCR recognition) → Decoder -> ...

    FEATURES:
    - Processes batches of cropped ROIs
    - Converts Texture → Tensor on GPU
    - "Latest batch wins" (older batches overwritten)
    - Sequential inference per ROI (no parallel GPU overload)
    - Async GPU readback (non-blocking)
    - Safe tensor ownership & disposal
*/
using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;

public class TextRecognitionInference : MonoBehaviour
{

    
    //---------- UI----------------
    [SerializeField] private ViewCroppedImage viewCroppedImage;

    private RenderTexture debugPreviewRT;
        [SerializeField]
    private int tensorTargetHeight = 48;

    [SerializeField]
    private int tensorTargetWidth = 320;

    [SerializeField] private Material cropMaterial;

    private RenderTexture croppedROI;

    private CommandBuffer commandBuffer;
     //------------------------------- 


    // ONNX OCR model (assigned in Inspector)
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private ProcessOCRDetection2 getROIText;

    /*
        Exact tensor input settings for ONNX model is:
        [DynamicDimension.0, 3, 48, DynamicDimension.1]
        in NCHW format.

        Therefore tensorTargetWidth can vary, though multiples of 32
        are recommended. Batch is still 1 per ROI.
    */

    // Loaded with the yml file
    [SerializeField]
    private TextAsset ymlFile;

    // Sentis worker → runs model on GPU
    private Worker worker;

    private TextDecoder textDecoder;

    private bool isProcessing = false;

    /*
        Holds latest incoming batch from ProcessOCRDetection.

        Each item = one frame's cropped ROI list + that frame data.

        IMPORTANT:
        - Overwritten on new incoming batch
        - Only latest batch is processed
    */
    private List<(List<(Tensor<float>, Rect)> rois, FrameData frame)> pendingBatch;

    //public event Action<Tensor<float>, FrameData> sendOCRTensor;

    private void Awake()
    {
        if (modelAsset == null || ymlFile == null || getROIText == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        var ocrModel = ModelLoader.Load(modelAsset);

        textDecoder = new TextDecoder(ymlFile);

        //CPU
        worker = new Worker(ocrModel, BackendType.CPU);

        //---------UI---------------
        debugPreviewRT = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        debugPreviewRT.Create();

        croppedROI = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        croppedROI.Create();

        commandBuffer = new CommandBuffer();
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
        Called when OCR detection post-processing produces a new batch.

        RESPONSIBILITIES:
        - Filter invalid textures
        - Copy batch safely
        - Store latest batch only
        - Start processing if not already running
    */
    private void onNewROI(List<(List<(Tensor<float>, Rect)>, FrameData)> roiBatch)
    {
        if (roiBatch == null || roiBatch.Count == 0)
        {
            return;
        }

        Debug.Log("new batch from ocr pp!");

        // Build safe filtered copy
        List<(List<(Tensor<float>, Rect)> rois, FrameData frame)> batch =
            new List<(List<(Tensor<float>, Rect)> rois, FrameData frame)>();

        PipelineProfiler.begin("Tensor filter 2");
        foreach (var item in roiBatch)
        {
            List<(Tensor<float>, Rect)> incomingRois = item.Item1;
            FrameData frame = item.Item2;

            if (incomingRois == null || incomingRois.Count == 0)
            {
                continue;
            }

            List<(Tensor<float>, Rect)> validRois = new List<(Tensor<float>, Rect)>();

            foreach (var roiEntry in incomingRois)
            {
                // GPT CODE
                Tensor<float> roiTensor = roiEntry.Item1;
                // GPT CODE
                Rect roiBounds = roiEntry.Item2;

                if (roiTensor != null)
                {
                    // GPT CODE
                    validRois.Add((roiTensor, roiBounds));
                }
            }

            if (validRois.Count > 0)
            {
                batch.Add((validRois, frame));
            }
        }
        PipelineProfiler.end("Tensor filter 2");

        if (batch.Count == 0)
        {
            return;
        }

        PipelineProfiler.begin("Batch disposal 2");
        // If an older batch was waiting but never processed, drop it safely
        if (pendingBatch != null)
        {
            DisposeNestedTensorBatch(pendingBatch);
        }
        PipelineProfiler.end("Batch disposal 2");

        // Overwrite previous batch ("latest wins")
        pendingBatch = batch;

        if (!isProcessing)
        {
            PipelineProfiler.begin("Process queue rec");
            StartCoroutine(ProcessQueue());
            PipelineProfiler.end("Process queue rec");
        }
    }

    private IEnumerator ProcessQueue()
    {
        isProcessing = true;

        while (pendingBatch != null)
        {
            // Save latest batch
            List<(List<(Tensor<float>, Rect)> rois, FrameData frame)> batch = pendingBatch;
            pendingBatch = null;

            foreach (var frameBatch in batch)
            {
                List<(Tensor<float>, Rect)> rois = frameBatch.rois;
                FrameData frame = frameBatch.frame;

                Debug.Log("Number of ROI this frame batch : " + rois.Count);

                foreach (var roiEntry in rois)
                {
                    // GPT CODE
                    Tensor<float> roiTensor = roiEntry.Item1;
                    // GPT CODE
                    Rect roiBounds = roiEntry.Item2;

                    if (roiTensor != null)
                    {
                        // GPT CODE
                        yield return runInference(roiTensor, frame, roiBounds);
                    }
                }
            }
        }

        isProcessing = false;
    }

    //Runs inference asynchronously.
    private IEnumerator runInference(Tensor<float> inputTensor, FrameData frame, Rect roiBounds)
    {
        if (inputTensor == null || worker == null)
        {
            yield return null;
            yield break;
        }

        PipelineProfiler.begin("OCR TextRecog");
        worker.Schedule(inputTensor);

        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>)
            .ReadbackAndCloneAsync()
            .GetAwaiter();

        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        PipelineProfiler.end("OCR TextRecog");

        // Dispose input tensor after inference has finished
        inputTensor.Dispose();

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            yield break;
        }

        PipelineProfiler.begin("Text decoder");

        string output = textDecoder.decode(outputTensor);

        PipelineProfiler.end("Text decoder");

        outputTensor.Dispose();
        Debug.Log($"Detected word is: {output} | Bounds: {roiBounds}");

        //-------------UI----------------
        TextureCropper.CropBoundingBox(roiBounds,frame.currentTexture,croppedROI,cropMaterial);
        Graphics.Blit(croppedROI, debugPreviewRT);
        viewCroppedImage.Show(debugPreviewRT); 
        //-------------UI----------------
    }

    private void DisposeNestedTensorBatch(List<(List<(Tensor<float>, Rect)> rois, FrameData frame)> batch)
    {
        if (batch == null)
        {
            return;
        }

        foreach (var frameBatch in batch)
        {
            if (frameBatch.rois == null)
            {
                continue;
            }

            foreach (var roiEntry in frameBatch.rois)
            {
                // GPT CODE
                roiEntry.Item1?.Dispose();
            }
        }
    }

    private void OnDestroy()
    {
        worker?.Dispose();

        if (pendingBatch != null)
        {
            DisposeNestedTensorBatch(pendingBatch);
            pendingBatch = null;
        }

        
        //----------UI------------
        if (debugPreviewRT != null)
        {
            debugPreviewRT.Release();
            Destroy(debugPreviewRT);
            debugPreviewRT = null;
        }


        if (commandBuffer != null)
        {
            commandBuffer.Release();
            commandBuffer = null;
        }

        if (croppedROI != null)
        {
            croppedROI.Release();
            Destroy(croppedROI);
            croppedROI = null;
        }

    }
}
