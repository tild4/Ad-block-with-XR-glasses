/// *Sammanfattning saknas*
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;

public class SentisInferenceManager : MonoBehaviour
{
    [SerializeField] private ModelAsset m_modelAsset; // *varibelnamn*
    [SerializeField, Range(0f, 1f)] private float m_confidenceThreshold = 0.5f; // *varibelnamn*
    [SerializeField] private CameraTextureToTensor m_cameraTextureToTensor; // *varibelnamn*
    // store the latest tensor and its timestamp received from CameraTextureToTensor
    private Tensor<float> m_latestTensor; // *varibelnamn*
    private DateTime m_latestTimestamp; // *varibelnamn*

    private Worker m_worker; // *varibelnamn*
    private Vector2Int m_inputSize; // *varibelnamn*

    public static List<(Rect boundingBox, float confidence, DateTime timestamp)> Detections { get; private set; } // *global static -> polling, skapa event istället?*
        = new List<(Rect, float, DateTime)>();

    private void Awake()
    {
        // Load the model from the assigned ModelAsset (the imported .onnx file)
        var model = ModelLoader.Load(m_modelAsset);

        // Read the expected input dimensions from the model itself (width and height)
        // model.inputs[0].shape is (1, 3, H, W) so index 2 = height, index 3 = width
        m_inputSize = new Vector2Int(model.inputs[0].shape.Get(2), model.inputs[0].shape.Get(3));

        // Create the worker - this is the engine that runs the model
        // BackendType.CPU means we run on the processor, not the GPU
        m_worker = new Worker(model, BackendType.CPU);
    }

    private void OnEnable()
    {
        // Subscribe to the GPU-converted tensor event from CameraTextureToTensor
        m_cameraTextureToTensor.sendTensor += OnNewTensor;
    }

    private void OnDisable()
    {
        // Unsubscribe when disabled to avoid memory leaks
        m_cameraTextureToTensor.sendTensor -= OnNewTensor;
    }

    private void OnNewTensor(Tensor<float> tensor, DateTime timestamp) // *metodnamn*
    {
        // store the latest GPU-converted tensor and its timestamp so RunInference() can use them
        m_latestTensor = tensor;
        m_latestTimestamp = timestamp;
    }

    private IEnumerator Start()
    {
        while (true)
        {
            yield return RunInference();
        }
    }

    private IEnumerator RunInference() // *metodnamn*
    {
        // check if we have received any tensor from CameraTextureToTensor yet
        if (m_latestTensor == null)
        {
            yield return null;
            yield break;
        }

        // snapshot the timestamp for this inference run
        DateTime frameTimestamp = m_latestTimestamp;

        // run model by scheduling all layers with the GPU-prepared tensor
        m_worker.Schedule(m_latestTensor);

        // Read the output tensor asynchronously so we dont freeze the main thread while waiting for results
        var outputAwaiter = (m_worker.PeekOutput(0) as Tensor<float>).ReadbackAndCloneAsync().GetAwaiter();

        // wait each frame until the output tensor is ready
        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        // get the output tensor result
        using var output = outputAwaiter.GetResult();

        if (output == null)
        {
            yield break;
        }

        // parse the output tensor of shape (1, 5, 8400):   YOLOv8 delar upp bilden i ett rutnät på tre olika skalor:
        // 640/8  = 80×80 = 6400 ankare   (hittar små objekt)
        // 640/16 = 40×40 = 1600 ankare   (hittar medelstora)
        // 640/32 = 20×20 =  400 ankare   (hittar stora)
        //                = 8400 totalt

        // clear previous frames detections
        Detections.Clear();

        // loop through all 8400 possible detctions
        int numDetections = output.shape[2];
        for (int i = 0; i < numDetections; i++)
        {
            // read confidence score for the detection
            float confidence = output[0, 4, i];

            // if confidence is below our threshold, skip this detection
            if (confidence < m_confidenceThreshold)
            {
                continue;
            }

            // read bounding box coordinates predicted by the model
            float centerX = output[0, 0, i];
            float centerY = output[0, 1, i];
            float width = output[0, 2, i];
            float height = output[0, 3, i];

            // convert from center + size to top-left corner + size, normalized to 0-1
            float x1 = (centerX - width / 2f) / m_inputSize.x;
            float y1 = (centerY - height / 2f) / m_inputSize.y;
            float nw = width / m_inputSize.x;
            float nh = height / m_inputSize.y;

            // store the results as a Rect (x, y, width, height), confidence score and frame timestamp
            Detections.Add((new Rect(x1, y1, nw, nh), confidence, frameTimestamp));
        }

        // Log detection results for debugging (visible via ADB logcat or Meta Quest Developer Hub)
        if (Detections.Count > 0)
            Debug.Log($"[Sentis] Detected {Detections.Count} ads. Top confidence: {Detections[0].confidence:0.00}");
        else
            Debug.Log("[Sentis] No ads detected this frame.");

        yield return null;
    }

    private void OnDestroy()
    {
        m_worker?.Dispose();
    }
}
