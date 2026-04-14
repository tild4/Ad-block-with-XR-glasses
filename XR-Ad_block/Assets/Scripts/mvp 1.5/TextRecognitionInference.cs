/*
    TextRecognitionInference

    PURPOSE:
    Runs OCR recognition on cropped word ROI tensors, decodes the text,
    and logs the recognized output.

    CURRENT FLOW:
    ProcessOCRDetection2 -> THIS

    POLICY:
    - Latest batch wins.
    - Older pending batches are disposed before they start.
    - Recognition runs sequentially per ROI.
*/

using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class TextRecognitionInference : MonoBehaviour
{
    [SerializeField]
    private ViewCroppedImage viewCroppedImage;

    [SerializeField]
    private ProcessOCRDetection getROIText;

    /*
    OCR recognition input:
    [DynamicDimension.0, 3, 48, DynamicDimension.1] in NCHW format.

    Width can vary, though multiples of 32 are recommended.
    Batch remains 1 per ROI.
    */

    [SerializeField]
    private int tensorTargetHeight = 48;

    [SerializeField]
    private int tensorTargetWidth = 320;

    /*
    [SerializeField]
    private int previewCropHeight = 128;

    [SerializeField]
    private int previewCropWidth = 512;
    */

    [SerializeField]
    private Material cropMaterial;

    // Reusable GPU resources
    private RenderTexture debugPreviewRT;
    private RenderTexture croppedROI;

    // REMOVE
    private CommandBuffer commandBuffer;

    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private TextAsset ymlFile;

    private Worker worker;
    private TextDecoder textDecoder;
    private bool isProcessing = false;

    // Each entry is 1 ad including all text inside it
    private List<FrameRoiBatch> pendingBatch;

    private void Awake()
    {
        if (modelAsset == null || ymlFile == null || getROIText == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        var ocrModel = ModelLoader.Load(modelAsset);

        textDecoder = new TextDecoder(ymlFile);
        worker = new Worker(ocrModel, BackendType.CPU);

        // CHANGE MAYBE
        debugPreviewRT = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );
        debugPreviewRT.Create();

        // CHANGE MAYBE
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
            getROIText.sendCroppedROIText += HandleNewRoiBatch;
        }
    }

    private void OnDisable()
    {
        if (getROIText != null)
        {
            getROIText.sendCroppedROIText -= HandleNewRoiBatch;
        }
    }

    /*
        Receives a new recognition batch.

        FLOW:
        1. Filter invalid ROI tensors.
        2. Keep only the newest pending batch.
        3. Start processing if needed.
    */
    private void HandleNewRoiBatch(List<FrameRoiBatch> roiBatch)
    {
        if (roiBatch == null || roiBatch.Count == 0)
        {
            return;
        }

        // Filters out null tensors
        List<FrameRoiBatch> batch = FilterValidRoiBatch(roiBatch);

        if (batch.Count == 0)
        {
            return;
        }

        ReplacePendingBatch(batch);

        if (!isProcessing)
        {
            StartCoroutine(ProcessQueue());
        }
    }

    private List<FrameRoiBatch> FilterValidRoiBatch(List<FrameRoiBatch> roiBatch)
    {
        List<FrameRoiBatch> batch = new List<FrameRoiBatch>();

        foreach (var item in roiBatch)
        {
            List<CroppedTextRoi> incomingRois = item.Rois;
            FrameData frame = item.Frame;

            if (incomingRois == null || incomingRois.Count == 0)
            {
                continue;
            }

            List<CroppedTextRoi> validRois = new List<CroppedTextRoi>();

            foreach (var roi in incomingRois)
            {
                if (roi.Tensor != null)
                {
                    validRois.Add(roi);
                }
            }

            if (validRois.Count > 0)
            {
                batch.Add(new FrameRoiBatch(validRois, frame));
            }
        }

        return batch;
    }

    /*
    If pending batch is not null -> it is old -> should be replaced
    Dispose tensors safely from the batch
    Replace pending batch with this one
    */
    private void ReplacePendingBatch(List<FrameRoiBatch> latestBatch)
    {
        PipelineProfiler.begin("Batch disposal 2");
        if (pendingBatch != null)
        {
            DisposeNestedTensorBatch(pendingBatch);
        }
        PipelineProfiler.end("Batch disposal 2");

        pendingBatch = latestBatch;
    }

    private IEnumerator ProcessQueue()
    {
        // Prevents nested coroutines
        isProcessing = true;

        // Loop is alive as long as there is work
        while (pendingBatch != null)
        {
            // Save latest batch
            List<FrameRoiBatch> batch = pendingBatch;
            pendingBatch = null;

            // For every list of text regions per detected ad
            foreach (var frameBatch in batch)
            {
                List<CroppedTextRoi> rois = frameBatch.Rois;
                FrameData frame = frameBatch.Frame;

                Debug.Log("Number of ROI this frame batch : " + rois.Count);

                // run inference for every text region
                foreach (var roi in rois)
                {
                    if (roi.Tensor != null)
                    {
                        yield return RunInference(roi.Tensor, frame, roi.Bounds);
                    }
                }
            }
        }

        isProcessing = false;
    }

    /*
        Runs OCR recognition for one cropped word ROI.

        FLOW:
        1. Schedule inference.
        2. Wait for async readback.
        3. Dispose the input tensor.
        4. Decode and log the recognized text.
    */
    private IEnumerator RunInference(Tensor<float> inputTensor, FrameData frame, Rect roiBounds)
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
        Debug.Log($"Detected word is: {output}");

        // UI code
        TextureCropper.CropBoundingBox(roiBounds, frame.currentTexture, croppedROI, cropMaterial);
        Graphics.Blit(croppedROI, debugPreviewRT);
        viewCroppedImage.Show(debugPreviewRT);
        viewCroppedImage.SetDetectedWord(output);
    }

    private void DisposeNestedTensorBatch(List<FrameRoiBatch> batch)
    {
        if (batch == null)
        {
            return;
        }

        foreach (var frameBatch in batch)
        {
            if (frameBatch.Rois == null)
            {
                continue;
            }

            foreach (var roi in frameBatch.Rois)
            {
                roi.Tensor?.Dispose();
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
