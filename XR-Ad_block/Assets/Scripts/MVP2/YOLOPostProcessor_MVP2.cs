// Copyright (c). All rights reserved.
//
// This file contains code inspired by samples from Meta Platforms, Inc.
// It is not a direct copy of Meta source; the implementation and
// copyright belong to this project unless otherwise stated.
//
/*
    YOLOPostProcessor

    This script processes raw YOLO detections from SentisInferenceManager and prepares them for OCR text detection.
    It performs the following steps:
    1. Converts raw detections into a structured format (DetectionData).
    2. Applies Non-Maximum Suppression (NMS) to filter overlapping detections.
    3. Crops the detected regions from the original frame, preserving aspect ratio.
    4. Converts the cropped regions into tensors suitable for OCR input.
    5. Emits the processed detections with associated tensors for downstream OCR processing.

    The script also includes debug functionality to preview cropped regions and adjust NMS thresholds.
*/
using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class YOLOPostProcessor_MVP2 : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private SentisInferenceManager sentisInferenceManager;

    [SerializeField]
    private ViewCroppedImage viewCroppedImage;

    private RenderTexture debugPreviewRT;

    [SerializeField]
    private Material cropMaterial;

    [Header("Post-Processing Settings")]
    [SerializeField, Range(0f, 1f)]
    private float iouThreshold = 0.4f;

    [Header("OCR Tensor Settings")]
    [SerializeField]
    private int tensorTargetHeight = 640;

    [SerializeField]
    private int tensorTargetWidth = 640;

    // List of processed detections ready for OCR, with cropped ROI tensors assigned
    private List<DetectionData> processedDetections = new List<DetectionData>();

    // Buffer list to avoid allocations when converting raw detections to DetectionData
    private List<DetectionData> detectionDataBuffer;

    // Reused GPU resources for tensor conversion.
    // NOTE: Crop currently uses temporary RenderTextures; 'croppedROI' is legacy (kept for compatibility) and not used by the current crop path.
    private RenderTexture croppedROI;
    private RenderTexture convertRenderTexture;
    private CommandBuffer commandBuffer;

    // Event sent to OCR text detection (invoked even when the list is empty).
    public event Action<List<DetectionData>> onProcessedDetections;

    /*
        Initialization:
        - Create necessary RenderTextures and CommandBuffer.
    */
    private void Awake()
    {
        if (sentisInferenceManager == null || cropMaterial == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        convertRenderTexture = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        croppedROI = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        convertRenderTexture.Create();
        croppedROI.Create();

        commandBuffer = new CommandBuffer();
        processedDetections = new List<DetectionData>();
        detectionDataBuffer = new List<DetectionData>();

        debugPreviewRT = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        debugPreviewRT.Create();
    }

    /*
        Subscribe to SentisInferenceManager's onDetectionsReady event to receive raw YOLO detections.
    */
    private void OnEnable()
    {
        if (sentisInferenceManager != null)
        {
            sentisInferenceManager.onDetectionsReady += HandleRawDetections;
        }
    }

    /*
        Unsubscribe from events to prevent memory leaks.
    */
    private void OnDisable()
    {
        if (sentisInferenceManager != null)
        {
            sentisInferenceManager.onDetectionsReady -= HandleRawDetections;
        }
    }

    /*
        Main handler for raw detections from YOLO:
        - Step 1: Convert raw detections to DetectionData list
        - Step 2: Apply Non-Maximum Suppression (NMS)
        - Step 3: Build processed detection batch with cropped ROI tensors
        - Step 4: Emit processed detections for OCR text detection
    */
    private void HandleRawDetections(
        List<(Rect boundingBox, float confidence, FrameData frame)> rawDetections
    )
    {
        processedDetections.Clear();
        // Early exit if no detections, but still invoke event to signal completion
        if (rawDetections == null || rawDetections.Count == 0)
        {
            onProcessedDetections?.Invoke(processedDetections);
            return;
        }
        // Step 1: Convert raw detections to DetectionData list
        List<DetectionData> detectionDataList = BuildDetectionDataList(rawDetections);
        // Step 2: Apply Non-Maximum Suppression (NMS)
        List<DetectionData> nmsResults = ApplyNMS(detectionDataList);
        // Step 3: Build processed detection batch with cropped ROI tensors
        BuildProcessedDetectionBatch(nmsResults);

        Debug.Log(
            $"Post-processed {rawDetections.Count} raw detections → {processedDetections.Count} final detections"
        );

        // Step 4: Emit processed detections for OCR text detection
        var sendDetections = new List<DetectionData>(processedDetections);
        onProcessedDetections?.Invoke(sendDetections);
    }

    /*
        Converts raw YOLO detections into structured DetectionData
    */
    private List<DetectionData> BuildDetectionDataList(
        List<(Rect boundingBox, float confidence, FrameData frame)> rawDetections
    )
    {
        detectionDataBuffer.Clear();
        // Convert each raw detection into DetectionData
        foreach (var (bbox, conf, frame) in rawDetections)
        {
            DetectionData data = new DetectionData
            {
                bboxNormalized = bbox,
                confidence = conf,
                frame = frame,
                RoiContentRectNormalized = new Rect(0f, 0f, 1f, 1f),
            };

            detectionDataBuffer.Add(data);
        }

        return detectionDataBuffer;
    }

    /*
    For each detected ad in the batch:
    1. Clamp Yolo bounds for safety
    2. Crop the region in the original frame using Yolo bounds
    3. Convert the cropped ROI to a tensor
    4. Add it to sending batch
    */
    private void BuildProcessedDetectionBatch(List<DetectionData> nmsResults)
    {
        Debug.Log("Number of items to be processed : " + nmsResults.Count);

        for (int i = 0; i < nmsResults.Count; i++)
        {
            // Work on a copy (struct), then assign tensor and add to processedDetections
            var result = nmsResults[i];

            Rect bbox = ClampNormalizedRect(result.bboxNormalized);
            Texture texture = result.frame.currentTexture;
            FrameData frame = result.frame;

            if (bbox.width <= 0f || bbox.height <= 0f)
            {
                continue;
            }

            // Crop at the ad's natural pixel resolution to preserve aspect ratio.
            // Previously we cropped into the fixed 640×640 croppedROI, which
            // stretched non-square ads and distorted the text (making it blurry
            // and harder for OCR to recognize). Now we crop at native resolution
            // first, then BlitWithAspectPad pads it into 640×640 with black bars.
            int naturalW = Mathf.Max(1, Mathf.RoundToInt(bbox.width * texture.width));
            int naturalH = Mathf.Max(1, Mathf.RoundToInt(bbox.height * texture.height));
            RenderTexture naturalCrop = RenderTexture.GetTemporary(
                naturalW,
                naturalH,
                0,
                RenderTextureFormat.ARGB32
            );

            if (!TextureCropper.CropBoundingBoxTopLeft(bbox, texture, naturalCrop, cropMaterial))
            {
                RenderTexture.ReleaseTemporary(naturalCrop);
                continue;
            }

            Graphics.Blit(naturalCrop, debugPreviewRT);
            //viewCroppedImage?.Show(debugPreviewRT);
            //viewCroppedImage?.SetDetectedWord(string.Empty);

            // Save a persistent copy of the ROI crop for OCR.
            // BlitWithAspectPad scales the
            // natural-resolution crop to fit inside 640×640 while preserving
            // aspect ratio, with black padding around the shorter dimension.
            RenderTexture snapshot = new RenderTexture(
                tensorTargetWidth,
                tensorTargetHeight,
                0,
                RenderTextureFormat.ARGB32
            );
            snapshot.Create();
            Rect contentRect = ConvertToTensor.BlitWithAspectPad(
                naturalCrop,
                snapshot,
                commandBuffer
            );
            RenderTexture.ReleaseTemporary(naturalCrop);

            PipelineProfiler.set("TensorContext", "YOLOPost");
            Tensor<float> roiTensor = ConvertToTensor.convert(
                snapshot,
                convertRenderTexture,
                tensorTargetHeight,
                tensorTargetWidth,
                commandBuffer
            );

            if (roiTensor != null)
            {
                result.bboxNormalized = bbox;
                result.RoiTensor = roiTensor;
                result.RoiSnapshot = snapshot;
                result.RoiContentRectNormalized = contentRect;
                processedDetections.Add(result);
            }
            else
            {
                snapshot.Release();
                Destroy(snapshot);
            }
        }
    }

    /*
        Non-Maximum Suppression (NMS):
        - Sort detections by confidence.
        - Keep the strongest box.
        - Suppress later boxes whose IOU exceeds the threshold.
    */
    private List<DetectionData> ApplyNMS(List<DetectionData> detections)
    {
        // Early exit if no detections
        if (detections.Count == 0)
        {
            return new List<DetectionData>();
        }
        // Sort detections by confidence in descending order
        detections.Sort((a, b) => b.confidence.CompareTo(a.confidence));

        List<DetectionData> results = new List<DetectionData>();
        bool[] suppressed = new bool[detections.Count];

        // Iterate through detections, applying NMS
        for (int i = 0; i < detections.Count; i++)
        {
            // Skip if this detection has already been suppressed
            if (suppressed[i])
            {
                continue;
            }

            results.Add(detections[i]);

            // Compare this detection with the remaining ones
            for (int j = i + 1; j < detections.Count; j++)
            {
                // Skip if the j-th detection has already been suppressed
                if (suppressed[j])
                {
                    continue;
                }
                // Calculate IOU between the current detection and the j-th detection
                float iou = CalculateIOU(
                    detections[i].bboxNormalized,
                    detections[j].bboxNormalized
                );
                // Suppress the j-th detection if IOU exceeds the threshold
                if (iou > iouThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        Debug.Log($"NMS: {detections.Count} detections → {results.Count} after suppression");
        return results;
    }

    /*
        Calculate Intersection over Union (IOU) between two rectangles.
    */
    private float CalculateIOU(Rect a, Rect b)
    {
        float xOverlap = Mathf.Max(0, Mathf.Min(a.xMax, b.xMax) - Mathf.Max(a.xMin, b.xMin));
        float yOverlap = Mathf.Max(0, Mathf.Min(a.yMax, b.yMax) - Mathf.Max(a.yMin, b.yMin));
        float intersectionArea = xOverlap * yOverlap;

        float unionArea = (a.width * a.height) + (b.width * b.height) - intersectionArea;
        return unionArea == 0 ? 0f : intersectionArea / unionArea;
    }

    /*
        Clamp normalized rectangle coordinates to ensure they are within [0, 1] range and have valid dimensions.
    */
    private Rect ClampNormalizedRect(Rect rect)
    {
        float xMin = Mathf.Clamp01(rect.xMin);
        float yMin = Mathf.Clamp01(rect.yMin);
        float xMax = Mathf.Clamp01(rect.xMax);
        float yMax = Mathf.Clamp01(rect.yMax);

        float width = xMax - xMin;
        float height = yMax - yMin;

        if (width <= 0f || height <= 0f)
        {
            return Rect.zero;
        }

        return new Rect(xMin, yMin, width, height);
    }

    /*
        Returns the currently configured IOU threshold used by NMS.
    */
    public float GetIouThreshold()
    {
        return iouThreshold;
    }

    /*
        Sets the IOU threshold used by NMS (clamped to [0,1]).
    */
    public void SetIouThreshold(float value)
    {
        iouThreshold = Mathf.Clamp01(value);
    }

    /*
        Clean up GPU resources to prevent memory leaks.
    */
    private void OnDestroy()
    {
        if (processedDetections != null)
        {
            foreach (var item in processedDetections)
            {
                item.RoiTensor?.Dispose();
                if (item.RoiSnapshot != null)
                {
                    item.RoiSnapshot.Release();
                    Destroy(item.RoiSnapshot);
                }
            }

            processedDetections.Clear();
        }

        if (commandBuffer != null)
        {
            commandBuffer.Release();
            commandBuffer = null;
        }

        if (convertRenderTexture != null)
        {
            convertRenderTexture.Release();
            Destroy(convertRenderTexture);
            convertRenderTexture = null;
        }

        if (croppedROI != null)
        {
            croppedROI.Release();
            Destroy(croppedROI);
            croppedROI = null;
        }

        if (debugPreviewRT != null)
        {
            debugPreviewRT.Release();
            Destroy(debugPreviewRT);
            debugPreviewRT = null;
        }
    }
}
