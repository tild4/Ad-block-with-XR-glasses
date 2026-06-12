/*
    Summary:
    Converts raw YOLO detections into OCR-ready DetectionData, including
    native-aspect ROI snapshots and tensors for text detection.

    Pipeline:
    YOLOInferenceManager -> YOLOPostProcessor -> TrackingManager

    Note:
    This project uses and adapts sample code provided through the Meta XR SDK.

    Copyright © Meta Platform Technologies, LLC and its affiliates.
    All rights reserved.
*/
using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class YOLOPostProcessor : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private YOLOInferenceManager yoloInferenceManager;

    [SerializeField]
    private Material cropMaterial;

    [Header("Post-Processing Settings")]
    [SerializeField, Range(0f, 1f)]
    private float iouThreshold = 0.4f;

    private const int TensorTargetHeight = 640;
    private const int TensorTargetWidth = 640;
    private List<DetectionData> processedDetections = new List<DetectionData>();
    private List<DetectionData> detectionDataBuffer;

    private RenderTexture convertRenderTexture;
    private CommandBuffer commandBuffer;

    public event Action<List<DetectionData>> onProcessedDetections;

    private void Awake()
    {
        if (yoloInferenceManager == null || cropMaterial == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        convertRenderTexture = new RenderTexture(
            TensorTargetWidth,
            TensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        convertRenderTexture.Create();

        commandBuffer = new CommandBuffer();
        processedDetections = new List<DetectionData>();
        detectionDataBuffer = new List<DetectionData>();
    }

    private void OnEnable()
    {
        if (yoloInferenceManager != null)
        {
            yoloInferenceManager.onDetectionsReady += HandleRawDetections;
        }
    }

    private void OnDisable()
    {
        if (yoloInferenceManager != null)
        {
            yoloInferenceManager.onDetectionsReady -= HandleRawDetections;
        }
    }

    private void HandleRawDetections(
        List<(Rect boundingBox, float confidence, FrameData frame)> rawDetections
    )
    {
        processedDetections.Clear();
        if (rawDetections == null || rawDetections.Count == 0)
        {
            onProcessedDetections?.Invoke(processedDetections);
            return;
        }

        List<DetectionData> detectionDataList = BuildDetectionDataList(rawDetections);
        List<DetectionData> nmsResults = ApplyNMS(detectionDataList);
        BuildProcessedDetectionBatch(nmsResults);

        Debug.Log(
            $"Post-processed {rawDetections.Count} raw detections -> {processedDetections.Count} final detections"
        );

        var sendDetections = new List<DetectionData>(processedDetections);
        onProcessedDetections?.Invoke(sendDetections);
    }

    private List<DetectionData> BuildDetectionDataList(
        List<(Rect boundingBox, float confidence, FrameData frame)> rawDetections
    )
    {
        detectionDataBuffer.Clear();
        foreach (var (bbox, conf, frame) in rawDetections)
        {
            DetectionData data = new DetectionData
            {
                bboxNormalized = bbox,
                confidence = conf,
                frame = frame,
            };

            detectionDataBuffer.Add(data);
        }

        return detectionDataBuffer;
    }

    private void BuildProcessedDetectionBatch(List<DetectionData> nmsResults)
    {
        Debug.Log("Number of items to be processed : " + nmsResults.Count);

        for (int i = 0; i < nmsResults.Count; i++)
        {
            var result = nmsResults[i];

            Rect bbox = ClampNormalizedRect(result.bboxNormalized);
            Texture texture = result.frame.currentTexture;
            FrameData frame = result.frame;

            if (bbox.width <= 0f || bbox.height <= 0f)
            {
                continue;
            }

            // Crop at native resolution first, then aspect-pad to the OCR tensor size.
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

            RenderTexture snapshot = new RenderTexture(
                TensorTargetWidth,
                TensorTargetHeight,
                0,
                RenderTextureFormat.ARGB32
            );
            snapshot.Create();
            ConvertToTensor.BlitWithAspectPad(naturalCrop, snapshot, commandBuffer);
            RenderTexture.ReleaseTemporary(naturalCrop);

            PipelineProfiler.set("TensorContext", "YOLOPost");
            PipelineProfiler.begin("OCR Prep 1 BGR ToTensor");
            Tensor<float> roiTensor = ConvertToTensor.convert(
                snapshot,
                convertRenderTexture,
                TensorTargetHeight,
                TensorTargetWidth,
                commandBuffer,
                ConvertToTensor.BgrChannelTransform
            );
            PipelineProfiler.end("OCR Prep 1 BGR ToTensor");

            if (roiTensor != null)
            {
                result.bboxNormalized = bbox;
                result.RoiTensor = roiTensor;
                result.RoiSnapshot = snapshot;
                processedDetections.Add(result);
            }
            else
            {
                snapshot.Release();
                Destroy(snapshot);
            }
        }
    }

    private List<DetectionData> ApplyNMS(List<DetectionData> detections)
    {
        if (detections.Count == 0)
        {
            return new List<DetectionData>();
        }

        detections.Sort((a, b) => b.confidence.CompareTo(a.confidence));

        List<DetectionData> results = new List<DetectionData>();
        bool[] suppressed = new bool[detections.Count];

        for (int i = 0; i < detections.Count; i++)
        {
            if (suppressed[i])
            {
                continue;
            }

            results.Add(detections[i]);

            for (int j = i + 1; j < detections.Count; j++)
            {
                if (suppressed[j])
                {
                    continue;
                }

                float iou = CalculateIOU(
                    detections[i].bboxNormalized,
                    detections[j].bboxNormalized
                );

                if (iou > iouThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        Debug.Log($"NMS: {detections.Count} detections -> {results.Count} after suppression");
        return results;
    }

    private float CalculateIOU(Rect a, Rect b)
    {
        float xOverlap = Mathf.Max(0, Mathf.Min(a.xMax, b.xMax) - Mathf.Max(a.xMin, b.xMin));
        float yOverlap = Mathf.Max(0, Mathf.Min(a.yMax, b.yMax) - Mathf.Max(a.yMin, b.yMin));
        float intersectionArea = xOverlap * yOverlap;

        float unionArea = (a.width * a.height) + (b.width * b.height) - intersectionArea;
        return unionArea == 0 ? 0f : intersectionArea / unionArea;
    }

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

    public float GetIouThreshold()
    {
        return iouThreshold;
    }

    public void SetIouThreshold(float value)
    {
        iouThreshold = Mathf.Clamp01(value);
    }

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
    }
}
