/*
    SentisInferenceManager

    PURPOSE:
    Runs YOLOv8 object detection model inference on camera frames.
    Converts camera texture to tensor using GPU, runs inference,
    and parses the output into bounding boxes.

    PIPELINE POSITION:
    CaptureCameraFrame → THIS (YOLO detection) → BlockPlacementManager

    FEATURES:
    - Converts camera Texture → Tensor on GPU via ConvertToTensor
    - Async GPU readback (non-blocking)
    - Processes only latest frame (no queue)
    - Safe tensor ownership & disposal
*/
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class SentisInferenceManager : MonoBehaviour
{
    // The imported ONNX model file, assigned in the Inspector
    [SerializeField]
    private ModelAsset modelAsset;

    // Minimum confidence score to count as a valid detection (0-1)
    [SerializeField, Range(0f, 1f)]
    private float confidenceThreshold = 0.5f;

    // Reference to CaptureCameraFrame, assigned in the Inspector
    [SerializeField]
    private CaptureCameraFrame captureCameraFrame;

    // The most recent GPU-converted tensor, owned by this class
    private Tensor<float> latestTensor;

    // The FrameData associated with the most recent tensor (pose, timestamp, etc.)
    private FrameData latestFrame;

    // The Sentis inference engine that executes the YOLO model
    private Worker worker;

    // The expected input dimensions read from the model (height, width)
    private Vector2Int inputSize;

    // Reused GPU resources for tensor conversion
    private RenderTexture renderTexture;
    private CommandBuffer commandBuffer;

    // Event fired after each inference run with all detections above the confidence threshold
    public event Action<
        List<(Rect boundingBox, float confidence, FrameData frame)>
    > onDetectionsReady;

    // Internal list of detections found in the current inference run
    private List<(Rect boundingBox, float confidence, FrameData frame)> detections =
        new List<(Rect, float, FrameData)>();

    private void Awake()
    {
        if (modelAsset == null || captureCameraFrame == null)
        {
            Debug.Log("missing asset");
            return;
        }

        // Load the YOLO model from the assigned ModelAsset
        var model = ModelLoader.Load(modelAsset);

        // Read expected input dimensions from the model: shape is (1, 3, H, W)
        inputSize = new Vector2Int(model.inputs[0].shape.Get(2), model.inputs[0].shape.Get(3));

        // Allocate reusable GPU resources for tensor conversion
        renderTexture = new RenderTexture(inputSize.y, inputSize.x, 0, RenderTextureFormat.ARGB32);
        renderTexture.Create();

        // CommandBuffer records GPU commands
        commandBuffer = new CommandBuffer();

        // Create the inference worker using CPU backend
        worker = new Worker(model, BackendType.CPU);
    }

    private void OnEnable()
    {
        // Subscribe to new camera frames
        if (captureCameraFrame != null)
        {
            captureCameraFrame.newFrame += onNewFrame;
        }
    }

    private void OnDisable()
    {
        // Unsubscribe to prevent memory leaks
        if (captureCameraFrame != null)
        {
            captureCameraFrame.newFrame -= onNewFrame;
        }
    }

    /*
        Called whenever a new camera frame is available.
        Converts the frame texture to a tensor on GPU.
        Disposes the previous tensor if it was never consumed by inference.
    */
    private void onNewFrame(FrameData frame)
    {
        // Skip conversion if inference hasn't consumed the previous tensor yet
        if (latestTensor != null)
        {
            return;
        }

        // Convert frame texture → tensor on GPU (this class owns the tensor)
        PipelineProfiler.set("TensorContext", "Sentis");
        latestTensor = ConvertToTensor.convert(
            frame.currentTexture,
            renderTexture,
            inputSize.x,
            inputSize.y,
            commandBuffer
        );

        latestFrame = frame;
    }

    // Continuously runs inference in a coroutine loop
    private IEnumerator Start()
    {
        while (true)
        {
            yield return runInference();
        }
    }

    /*
        Runs one inference pass.

        FLOW:
        1. Transfer tensor ownership from latestTensor to local variable
        2. Run GPU inference
        3. Await async readback
        4. Dispose input tensor
        5. Parse detections
    */
    private IEnumerator runInference()
    {
        // Wait until we have received at least one tensor
        if (latestTensor == null)
        {
            yield return null;
            yield break;
        }

        // Snapshot the FrameData so it stays consistent even if a new tensor arrives during inference
        FrameData frame = latestFrame;

        // Transfer ownership safely: inputTensor takes over, latestTensor is cleared
        Tensor<float> inputTensor = latestTensor;
        latestTensor = null;

        // Feed the tensor into the YOLO model
        PipelineProfiler.begin("3. AI Inference");
        PipelineProfiler.set("Model Input", $"{inputSize.x}x{inputSize.y}");
        worker.Schedule(inputTensor);

        // Start async readback of the output tensor to avoid blocking the main thread
        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>)
            .ReadbackAndCloneAsync()
            .GetAwaiter();

        // Yield each frame until the async readback is complete
        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        PipelineProfiler.end("3. AI Inference");

        // Dispose input tensor after inference (consumer responsibility)
        inputTensor.Dispose();

        // Retrieve the output tensor (shape: 1, 5, 8400)
        using var output = outputAwaiter.GetResult();

        // If readback failed, skip this frame
        if (output == null)
        {
            yield break;
        }

        // Output shape (1, 5, 8400):
        // YOLOv8 splits the image into a grid at three scales:
        // 640/8  = 80x80 = 6400 anchors (detects small objects)
        // 640/16 = 40x40 = 1600 anchors (detects medium objects)
        // 640/32 = 20x20 =  400 anchors (detects large objects)
        //                 = 8400 total
        // For each anchor: [centerX, centerY, width, height, confidence]

        // Clear detections from the previous inference run
        PipelineProfiler.begin("4. Post-Processing (Boxes)");
        detections.Clear();

        // Total number of anchor predictions in the output
        int numDetections = output.shape[2];

        // Loop through all anchor predictions
        for (int i = 0; i < numDetections; i++)
        {
            // Read the confidence score for this anchor
            float confidence = output[0, 4, i];

            // Skip if below the confidence threshold
            if (confidence < confidenceThreshold)
            {
                continue;
            }

            // Read bounding box in pixel coordinates (relative to 640x640 input)
            float centerX = output[0, 0, i];
            float centerY = output[0, 1, i];
            float width = output[0, 2, i];
            float height = output[0, 3, i];

            // Convert from center+size to top-left corner+size, normalized to 0-1
            float x1 = (centerX - width / 2f) / inputSize.x;
            float y1 = (centerY - height / 2f) / inputSize.y;
            float normalizedWidth = width / inputSize.x;
            float normalizedHeight = height / inputSize.y;

            // Store the detection with its bounding box, confidence, and capture-time FrameData
            detections.Add(
                (new Rect(x1, y1, normalizedWidth, normalizedHeight), confidence, frame)
            );
        }

        PipelineProfiler.end("4. Post-Processing (Boxes)");
        PipelineProfiler.set("Detections", detections.Count);

        // Log results for debugging (visible via ADB logcat or Meta Quest Developer Hub)
        if (detections.Count > 0)
        {
            Debug.Log(
                $"[Sentis] Detected {detections.Count} ads. Top confidence: {detections[0].confidence:0.00}"
            );

            var sendDetections = new List<(Rect boundingBox, float confidence, FrameData frame)>(
                detections
            );
            onDetectionsReady?.Invoke(sendDetections);
            Debug.Log("sent event!");
        }
        else
        {
            Debug.Log("[Sentis] No ads detected this frame.");
        }

        yield return null;
    }

    // cleanup of GPU resources and native memory
    private void OnDestroy()
    {
        // Dispose any unconsumed tensor
        latestTensor?.Dispose();

        // Release the inference worker
        worker?.Dispose();

        // Release GPU command buffer
        if (commandBuffer != null)
        {
            commandBuffer.Release();
            commandBuffer = null;
        }

        // Release GPU render texture
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
            renderTexture = null;
        }
    }
}
