/*
    YOLOPostProcessor

    PURPOSE:
    Receives raw YOLO detections, removes overlapping boxes with NMS,
    crops each surviving ROI, converts it to a tensor, and forwards the
    batch to OCR text detection.

    CURRENT FLOW:
    SentisInferenceManager -> THIS -> TextDetectionInference

    NOTE: 
    Emitted tensor is the cropped ROI of the ad
*/

using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;
public class YOLOPostProcessor_uml : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private SentisInferenceManager sentisInferenceManager;

    /*
    ============================== UI ==============================
    [SerializeField] private ViewCroppedImage viewCroppedImage;
    private RenderTexture debugPreviewRT;
    ===================================================================
    */

    [SerializeField]
    private Material cropMaterial;

    [Header("Post-Processing Settings")]

    [SerializeField, Range(0f, 1f)]
    private float iouThreshold = 0.4f;

    [SerializeField]
    private int tensorTargetHeight = 640;

    [SerializeField]
    private int tensorTargetWidth = 640;

    // Each entry is a tensor of 1 ad
    private readonly List<YoloRoiTensor> processedDetections = new List<YoloRoiTensor>();

    // Reused GPU resources for crop + tensor conversion.
    private RenderTexture croppedROI;
    private RenderTexture convertRenderTexture;
    private CommandBuffer commandBuffer;

    // Event sent to OCR text detection.
    public event Action<List<YoloRoiTensor>> onProcessedDetections;

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

        /*
        ============================== UI ==============================
        debugPreviewRT = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        debugPreviewRT.Create();
        ===================================================================
        */
    }

    private void OnEnable()
    {
        if (sentisInferenceManager != null)
        {
            sentisInferenceManager.onDetectionsReady += HandleRawDetections;
        }
    }

    private void OnDisable()
    {
        if (sentisInferenceManager != null)
        {
            sentisInferenceManager.onDetectionsReady -= HandleRawDetections;
        }
    }

    /*
        Handles one raw YOLO detection batch.

        FLOW:
        1. Convert raw detections into DetectionData.
        2. Apply NMS.
        3. Crop and tensorize each surviving ROI.
        4. Emit the processed ROI tensor batch.
    */
    private void HandleRawDetections(
        List<(Rect boundingBox, float confidence, FrameData frame)> rawDetections
    )
    {
        processedDetections.Clear();

        if (rawDetections == null || rawDetections.Count == 0)
        {
            return;
        }

        List<DetectionData> detectionDataList = BuildDetectionDataList(rawDetections);
        List<DetectionData> nmsResults = ApplyNMS(detectionDataList);
        BuildProcessedDetectionBatch(nmsResults);

        Debug.Log(
            $"Post-processed {rawDetections.Count} raw detections → {processedDetections.Count} final detections"
        );

        // Send a copy of the list to avoid mutations and referencing issues
        var sendDetections = new List<YoloRoiTensor>(processedDetections);

        onProcessedDetections?.Invoke(sendDetections);
    }

    private List<DetectionData> BuildDetectionDataList(
        List<(Rect boundingBox, float confidence, FrameData frame)> rawDetections
    )
    {
        List<DetectionData> detectionDataList = new List<DetectionData>();

        foreach (var (bbox, conf, frame) in rawDetections)
        {
            DetectionData data = new DetectionData
            {
                bboxNormalized = bbox,
                bboxPixels = ConvertToPixelCoordinates(bbox, frame.currentResolution),
                confidence = conf,
                frame = frame,
            };

            detectionDataList.Add(data);
        }

        return detectionDataList;
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
        foreach (var result in nmsResults)
        {
            Rect bbox = ClampNormalizedRect(result.bboxNormalized);
            Texture texture = result.frame.currentTexture;
            FrameData frame = result.frame;

            if (bbox.width <= 0f || bbox.height <= 0f)
            {
                continue;
            }


            if (!TextureCropper.CropBoundingBox(bbox, texture, croppedROI, cropMaterial))
            {
                continue;
            }

            /*
            ============================== DELETE ==============================
            Graphics.Blit(croppedROI, debugPreviewRT);
            viewCroppedImage.Show(debugPreviewRT);
            ===================================================================
            */

            Tensor<float> roiTensor = ConvertToTensor.convert(
                croppedROI,
                convertRenderTexture,
                tensorTargetHeight,
                tensorTargetWidth,
                commandBuffer
            );

            // NOTE : bbox is the Yolo bounds
            if (roiTensor != null)
            {
                processedDetections.Add(new YoloRoiTensor(roiTensor, frame, bbox));
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

        Debug.Log($"NMS: {detections.Count} detections → {results.Count} after suppression");
        return results;
    }

    private float CalculateIOU(Rect a, Rect b)
    {
        float xOverlap = Mathf.Max(0, Mathf.Min(a.xMax, b.xMax) - Mathf.Max(a.xMin, b.xMin));
        float yOverlap = Mathf.Max(0, Mathf.Min(a.yMax, b.yMax) - Mathf.Max(a.yMin, b.yMin));
        float intersectionArea = xOverlap * yOverlap;

        float aArea = a.width * a.height;
        float bArea = b.width * b.height;
        float unionArea = aArea + bArea - intersectionArea;

        if (unionArea == 0)
        {
            return 0f;
        }

        return intersectionArea / unionArea;
    }

    private Rect ConvertToPixelCoordinates(Rect normalized, Vector2Int resolution)
    {
        return new Rect(
            normalized.x * resolution.x,
            normalized.y * resolution.y,
            normalized.width * resolution.x,
            normalized.height * resolution.y
        );
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
                item.Tensor?.Dispose();
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

        /*
        ============================== UI ==============================
        if (debugPreviewRT != null)
        {
            debugPreviewRT.Release();
            Destroy(debugPreviewRT);
            debugPreviewRT = null;
        }
        ===================================================================
        */
    }
}
