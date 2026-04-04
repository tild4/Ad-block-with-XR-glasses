/*
    DetectionPostProcessor

    PURPOSE:
    Filters and refines raw YOLO detections to ensure only high-quality,
    unique bounding boxes are passed further into the system.

    ARCHITECTURE:
    - Data Filtering: Removes detections with confidence scores below 'confidenceThreshold'.
    - Non-Maximum Suppression (NMS): An algorithm that identifies overlapping boxes
      and keeps only the one with the highest confidence, using 'iouThreshold' (Intersection over Union).
    - Coordinate Transformation: Converts normalized (0-1) AI coordinates into
      actual pixel coordinates based on the frame resolution.
    - Event-Driven: Listens to 'sentisInferenceManager' and broadcasts 'onProcessedDetections'.

    IMPORTANT:
    - Pre-allocates lists in Awake to avoid Garbage Collection (GC) spikes and
      maintain high performance during AR tracking.
    - IOU calculation is critical: if set too low, you lose valid detections;
      if set too high, you get duplicate blocks.
*/

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.InferenceEngine;

public class YOLOPostProcessor : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private SentisInferenceManager sentisInferenceManager;

    /* ---------- UI----------------
    [SerializeField] private ViewCroppedImage viewCroppedImage;
    private RenderTexture debugPreviewRT;
     ------------------------------- */

    [SerializeField] private Material cropMaterial;

    [Header("Post-Processing Settings")]
    
    /*
    [SerializeField, Range(0f, 1f)]
    private float confidenceThreshold = 0.5f;
    */

    [SerializeField, Range(0f, 1f)]
    private float iouThreshold = 0.4f;

    [SerializeField]
    private int tensorTargetHeight = 640;

    [SerializeField]
    private int tensorTargetWidth = 640;

    // Internal list for processed detections (reused to avoid allocations)
    private List<(Tensor<float>, FrameData, Rect)> processedDetections = new List<(Tensor<float>, FrameData, Rect)>();

    // Reused GPU resources for conversion
    private RenderTexture croppedROI;

    private RenderTexture convertRenderTexture;

    private CommandBuffer commandBuffer;

    // Event to send processed detections
    public event Action<List<(Tensor<float>, FrameData, Rect)>> onProcessedDetections;


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

        //---------UI---------------
        /*
        debugPreviewRT = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        debugPreviewRT.Create();
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
        Main handler for raw detections from SentisInferenceManager.
        Steps:
        1. Filter by confidence threshold.
        2. Convert to DetectionData format (normalized + pixel coordinates).
        3. Apply Non-Maximum Suppression (NMS) to remove duplicates.
        4. Store results and notify subscribers via onProcessedDetections event.
    */
    private void HandleRawDetections(
        List<(Rect boundingBox, float confidence, FrameData frame)> rawDetections
    )
    {
        // Clear previous results
        processedDetections.Clear();

        if (rawDetections == null || rawDetections.Count == 0)
        {
            return;
        }

        /*
        // 1. Filter by confidence threshold
        List<(Rect, float, FrameData)> filtered = new List<(Rect, float, FrameData)>();

        foreach (var detection in rawDetections)
        {
            if (detection.confidence >= confidenceThreshold)
            {
                filtered.Add(detection);
            }
        }

        if (filtered.Count == 0)
        {
            return;
        }
        */

        // 2. Convert to DetectionData format
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

        PipelineProfiler.begin("Apply nms on batch");
        
        // 3. Apply Non-Maximum Suppression
        List<DetectionData> nmsResults = ApplyNMS(detectionDataList);
        
        PipelineProfiler.end("Apply nms on batch");

        PipelineProfiler.begin("Handle filtered batch");

        Debug.Log("Number of items to be processed : " + nmsResults.Count);

        foreach(var result in nmsResults)
        {
            
            //var bbox = result.bboxNormalized;
            Rect bbox = ClampNormalizedRect(result.bboxNormalized);
            var texture = result.frame.currentTexture;
            var frame = result.frame;

            // Skip boxes that became invalid after clamping TMP
            if (bbox.width <= 0f || bbox.height <= 0f)
            {
                continue;
            }


            PipelineProfiler.begin("Crop time");

            if (!TextureCropper.CropBoundingBox(bbox, texture, croppedROI, cropMaterial))
            {
                continue;
            }

            PipelineProfiler.end("Crop time");

            //----------------UI-------------------
            /*
            // Copy the cropped result into a dedicated preview RT
            Graphics.Blit(croppedROI, debugPreviewRT);
            viewCroppedImage.Show(debugPreviewRT);     
            */

            Tensor<float> roiTensor = ConvertToTensor.convert(croppedROI,convertRenderTexture,tensorTargetHeight,tensorTargetWidth,commandBuffer);

            if (roiTensor != null)
            {
                processedDetections.Add((roiTensor, frame, bbox));               
            }
        }

        PipelineProfiler.end("Handle filtered batch");

        // 5. Notify subscribers
        Debug.Log(
            $"Post-processed {rawDetections.Count} raw detections → {processedDetections.Count} final detections"
        );

        var sendDetections = new List<(Tensor<float>, FrameData, Rect)>(processedDetections);

        onProcessedDetections?.Invoke(sendDetections);
    }

    /*
        Non-Maximum Suppression (NMS) algorithm:
        - Sorts detections by confidence score (highest first).
        - Iteratively selects the highest confidence detection and suppresses
          any remaining detections that have an IOU above 'iouThreshold' with it.
        - Returns a list of unique, high-confidence detections.
    */
    private List<DetectionData> ApplyNMS(List<DetectionData> detections)
    {
        if (detections.Count == 0)
        {
            return new List<DetectionData>();
        }

        // Sort by confidence (highest first)
        detections.Sort((a, b) => b.confidence.CompareTo(a.confidence));

        List<DetectionData> results = new List<DetectionData>();
        bool[] suppressed = new bool[detections.Count];

        for (int i = 0; i < detections.Count; i++)
        {
            if (suppressed[i])
            {
                continue;
            }

            // Keep this detection
            results.Add(detections[i]);

            // Suppress overlapping detections
            for (int j = i + 1; j < detections.Count; j++)
            {
                if (suppressed[j])
                {
                    continue;
                }

                // Calculate IOU
                float iou = CalculateIOU(
                    detections[i].bboxNormalized,
                    detections[j].bboxNormalized
                );

                // If IOU is above threshold, suppress the lower confidence detection
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
        Calculates Intersection over Union (IOU) between two bounding boxes.
        IOU = (Area of Intersection) / (Area of Union)
        Returns a value between 0 and 1, where 1 means perfect overlap.
    */
    private float CalculateIOU(Rect a, Rect b)
    {
        // Calculate intersection
        float xOverlap = Mathf.Max(0, Mathf.Min(a.xMax, b.xMax) - Mathf.Max(a.xMin, b.xMin));
        float yOverlap = Mathf.Max(0, Mathf.Min(a.yMax, b.yMax) - Mathf.Max(a.yMin, b.yMin));
        float intersectionArea = xOverlap * yOverlap;

        // Calculate union
        float aArea = a.width * a.height;
        float bArea = b.width * b.height;
        float unionArea = aArea + bArea - intersectionArea;

        // Avoid division by zero
        if (unionArea == 0)
        {
            return 0f;
        }

        return intersectionArea / unionArea;
    }

    /*
        Converts normalized bounding box coordinates (0-1) to pixel coordinates based on frame resolution.
        This is necessary for accurate rendering and interaction in the AR environment.
    */
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

        // Reject fully collapsed / invalid boxes after clamping
        if (width <= 0f || height <= 0f)
        {
            return Rect.zero;
        }

        return new Rect(xMin, yMin, width, height);
    }

    // Public getters for debugging/UI


    /*
    public float GetConfidenceThreshold()
    {
        return confidenceThreshold;
    }

    public void SetConfidenceThreshold(float value)
    {
        confidenceThreshold = Mathf.Clamp01(value);
    }
    */

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
    // Dispose any ROI tensors still stored locally
    if (processedDetections != null)
    {
        foreach (var item in processedDetections)
        {
            item.Item1?.Dispose();
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

    //----------UI------------
    /*

    if (debugPreviewRT != null)
    {
        debugPreviewRT.Release();
        Destroy(debugPreviewRT);
        debugPreviewRT = null;
    }
    */
}
}
