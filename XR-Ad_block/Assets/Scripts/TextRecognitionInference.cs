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
    // ONNX OCR model (assigned in Inspector)
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private ProcessOCRDetection getROIText;

    /*
        Exact tensor input settings for ONNX model is:
        [DynamicDimension.0, 3, 48, DynamicDimension.1]
        in NCHW format.

        Therefore tensorTargetWidth can vary, though multiples of 32
        are recommended. Batch is still 1 per ROI.
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

    /*
        Holds latest incoming batch from ProcessOCRDetection.

        Each item = one frame's cropped ROI list + that frame data.

        IMPORTANT:
        - Overwritten on new incoming batch
        - Only latest batch is processed
    */
    private List<(List<Texture> rois, FrameData frame)> pendingBatch;

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
        Called when OCR detection post-processing produces a new batch.

        RESPONSIBILITIES:
        - Filter invalid textures
        - Copy batch safely
        - Store latest batch only
        - Start processing if not already running
    */
    private void onNewROI(List<(List<Texture>, FrameData)> roiBatch)
    {
        if (roiBatch == null || roiBatch.Count == 0)
        {
            return;
        }

        // Build safe filtered copy
        List<(List<Texture> rois, FrameData frame)> batch = new List<(List<Texture> rois, FrameData frame)>();

        foreach (var item in roiBatch)
        {
            List<Texture> incomingRois = item.Item1;
            FrameData frame = item.Item2;

            if (incomingRois == null || incomingRois.Count == 0)
            {
                continue;
            }

            List<Texture> validRois = new List<Texture>();

            foreach (Texture roi in incomingRois)
            {
                if (roi != null)
                {
                    validRois.Add(roi);
                }
            }

            if (validRois.Count > 0)
            {
                batch.Add((validRois, frame));
            }
        }

        if (batch.Count == 0)
        {
            return;
        }


        if (pendingBatch != null)
        {
            foreach (var oldFrameBatch in pendingBatch)
            {
                foreach (Texture oldRoi in oldFrameBatch.rois)
                {
                    if (oldRoi is Texture2D oldRoi2D)
                    {
                        Destroy(oldRoi2D);
                    }
                }
            }
        }

        // Optional debug preview
        if (viewCroppedImage != null && batch[0].rois.Count > 0)
        {
            viewCroppedImage.Show(batch[0].rois[0]);
        }

        // Overwrite previous batch ("latest wins")
        pendingBatch = batch;

        if (!isProcessing)
        {
            StartCoroutine(ProcessQueue());
        }
    }

    private IEnumerator ProcessQueue()
    {
        isProcessing = true;

        while (pendingBatch != null)
        {
            // Save latest batch
            List<(List<Texture> rois, FrameData frame)> batch = pendingBatch;
            pendingBatch = null;

            foreach (var frameBatch in batch)
            {
                List<Texture> rois = frameBatch.rois;
                FrameData frame = frameBatch.frame;

                Debug.Log("Number of ROI this frame batch : " + rois.Count);

                foreach (Texture roi in rois)
                {
                    if (roi == null)
                    {
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
                    }

                    if (roi is Texture2D roi2D)
                    {
                        Destroy(roi2D);
                    }
                }
            }
        }

        isProcessing = false;
    }

    //Runs inference asynchronously.
    private IEnumerator runInference(Tensor<float> inputTensor, FrameData frame)
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
        Debug.Log("Detected word is: " + output);
    }

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

        if (pendingBatch != null)
        {
            foreach (var frameBatch in pendingBatch)
            {
                foreach (Texture roi in frameBatch.rois)
                {
                    if (roi is Texture2D roi2D)
                    {
                        Destroy(roi2D);
                    }
                }
            }

            pendingBatch = null;
        }
    }
}
