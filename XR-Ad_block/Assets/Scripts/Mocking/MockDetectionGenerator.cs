using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Simulates YOLO detections for testing without headset.
/// </summary>
public class MockDetectionGenerator : MonoBehaviour
{
    [SerializeField] private TrackingManager trackingManager;
    
    [Header("Mock Settings")]
    [SerializeField] private float detectionInterval = 2.0f;
    [SerializeField] private int numDetections = 2;
    
    private float timer = 0f;
    
    private void Update()
    {
        timer += Time.deltaTime;
        
        if (timer >= detectionInterval)
        {
            timer = 0f;
            SendMockDetection();
        }
    }
    
    private void SendMockDetection()
    {
        List<DetectionData> mockDetections = new List<DetectionData>();
        
        for (int i = 0; i < numDetections; i++)
        {
            Rect bbox = new Rect(
                Random.Range(0.1f, 0.5f),
                Random.Range(0.1f, 0.5f),
                Random.Range(0.1f, 0.3f),
                Random.Range(0.1f, 0.3f)
            );
            
            DetectionData mockData = new DetectionData
            {
                bboxNormalized = bbox,
                bboxPixels = new Rect(
                    bbox.x * 1920,
                    bbox.y * 1080,
                    bbox.width * 1920,
                    bbox.height * 1080
                ),
                confidence = Random.Range(0.6f, 0.95f),
                frame = CreateMockFrameData()
            };
            
            mockDetections.Add(mockData);
        }
        
        Debug.Log($"[MOCK] Sending {mockDetections.Count} fake detections");
        
        // Trigger TrackingManager directly (bypass DetectionPostProcessor)
        if (trackingManager != null)
        {
            // Access the private method via reflection
            var method = typeof(TrackingManager).GetMethod("UpdateDetections", 
                System.Reflection.BindingFlags.Instance | 
                System.Reflection.BindingFlags.NonPublic);
            
            if (method != null)
            {
                method.Invoke(trackingManager, new object[] { mockDetections });
            }
        }
    }
    
    private FrameData CreateMockFrameData()
    {
        return new FrameData
        {
            currentTexture = null,
            currentPose = new Pose(Vector3.zero, Quaternion.identity),
            currentRay = new Ray(Vector3.zero, Vector3.forward),
            currentResolution = new Vector2Int(1920, 1080),
            currentTimestamp = System.DateTime.Now
        };
    }
}