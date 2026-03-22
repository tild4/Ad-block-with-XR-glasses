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

public class DetectionPostProcessor : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField] private SentisInferenceManager sentisInferenceManager;
    
    [Header("Post-Processing Settings")]
    [SerializeField, Range(0f, 1f)] 
    private float confidenceThreshold = 0.5f;
    
    [SerializeField, Range(0f, 1f)] 
    private float iouThreshold = 0.4f;
    
    // Internal list for processed detections (reused to avoid allocations)
    private List<DetectionData> processedDetections = new List<DetectionData>();
    
    // Event to send processed detections
    public event Action<List<DetectionData>> onProcessedDetections;
    
    private void Awake()
    {
        // Pre-allocate list to avoid GC during runtime
        processedDetections = new List<DetectionData>();
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
    private void HandleRawDetections(List<(Rect boundingBox, float confidence, FrameData frame)> rawDetections)
    {
        // Clear previous results
        processedDetections.Clear();
        
        if (rawDetections == null || rawDetections.Count == 0)
        {
            // No detections - still notify with empty list
            onProcessedDetections?.Invoke(processedDetections);
            return;
        }
        
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
            onProcessedDetections?.Invoke(processedDetections);
            return;
        }
        
        // 2. Convert to DetectionData format
        List<DetectionData> detectionDataList = new List<DetectionData>();
        
        foreach (var (bbox, conf, frame) in filtered)
        {
            DetectionData data = new DetectionData
            {
                bboxNormalized = bbox,
                bboxPixels = ConvertToPixelCoordinates(bbox, frame.currentResolution),
                confidence = conf,
                frame = frame
            };
            
            detectionDataList.Add(data);
        }
        
        // 3. Apply Non-Maximum Suppression
        List<DetectionData> nmsResults = ApplyNMS(detectionDataList);
        
        // 4. Store results
        processedDetections.AddRange(nmsResults);
        
        // 5. Notify subscribers
        Debug.Log($"Post-processed {rawDetections.Count} raw detections → {processedDetections.Count} final detections");
        
        onProcessedDetections?.Invoke(processedDetections);
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
    
    // Public getters for debugging/UI
    
    public float GetConfidenceThreshold()
    {
        return confidenceThreshold;
    }
    
    public void SetConfidenceThreshold(float value)
    {
        confidenceThreshold = Mathf.Clamp01(value);
    }
    
    public float GetIouThreshold()
    {
        return iouThreshold;
    }
    
    public void SetIouThreshold(float value)
    {
        iouThreshold = Mathf.Clamp01(value);
    }
}