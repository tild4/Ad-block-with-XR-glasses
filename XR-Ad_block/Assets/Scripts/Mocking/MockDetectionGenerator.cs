/*
MockDetectionGenerator

PURPOSE:
Enables development and debugging in the Unity Editor without the need for
Meta Quest hardware or a real AI inference stream.

ARCHITECTURE:
- Automated Testing: Periodically generates randomized 2D bounding boxes
  simulating successful YOLO detections.
- System Integration: Uses C# Reflection to inject mock data directly into
  the TrackingManager's private 'UpdateDetections' method. This allows
  testing of the entire tracking and placement logic while bypassing
  the actual AI inference and Post-Processor.
- Frame Simulation: Constructs dummy 'FrameData' objects to satisfy the
  requirements of the tracking pipeline.

IMPORTANT:
This component is strictly for development and should be disabled or
removed before building for the Quest hardware.
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MockDetectionGenerator : MonoBehaviour
{
    [SerializeField]
    private TrackingManager trackingManager;

    [Header("Mock Settings")]
    [SerializeField]
    private float detectionInterval = 2.0f; // Time in seconds between each mock detection generation

    [SerializeField]
    private int numDetections = 2; // Number of mock detections to generate each time

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

    /*
        Creates a list of DetectionData objects with randomized
        bounding boxes and confidence values
    */
    private void SendMockDetection()
    {
        List<DetectionData> mockDetections = new List<DetectionData>();

        // Generate random detections
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
                frame = CreateMockFrameData(),
            };

            mockDetections.Add(mockData);
        }

        Debug.Log($"[MOCK] Sending {mockDetections.Count} fake detections");

        // Trigger TrackingManager directly (bypass DetectionPostProcessor)
        if (trackingManager != null)
        {
            // Access the private method via reflection
            var method = typeof(TrackingManager).GetMethod(
                "UpdateDetections",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic
            );

            if (method != null)
            {
                method.Invoke(trackingManager, new object[] { mockDetections });
            }
        }
    }

    /*
        Creates a dummy FrameData object with default values.
    */
    private FrameData CreateMockFrameData()
    {
        return new FrameData
        {
            currentTexture = null,
            currentPose = new Pose(Vector3.zero, Quaternion.identity),
            currentRay = new Ray(Vector3.zero, Vector3.forward),
            currentResolution = new Vector2Int(1920, 1080),
            currentTimestamp = System.DateTime.Now,
        };
    }
}
