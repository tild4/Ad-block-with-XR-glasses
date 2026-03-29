/// Runs YOLOv8 object detection on GPU-prepared tensors from CameraTextureToTensor.
/// Parses the model output into bounding boxes and fires an event with the detections
/// so that downstream scripts (e.g. BlockPlacementManager) can subscribe and react.
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

public class SentisInferenceManager : MonoBehaviour
{
    // The imported ONNX model file, assigned in the Inspector
    [SerializeField]
    private ModelAsset modelAsset;

    // Minimum confidence score to count as a valid detection (0-1)
    [SerializeField, Range(0f, 1f)]
    private float confidenceThreshold = 0.5f;

    // Reference to CameraTextureToTensor, assigned in the Inspector
    [SerializeField]
    private CameraTextureToTensor cameraTextureToTensor;

    [SerializeField] private ViewCroppedImage viewCroppedImage;

    // The most recent GPU-prepared tensor received from CameraTextureToTensor
    private Tensor<float> latestTensor;

    // The FrameData associated with the most recent tensor (pose, timestamp, etc.)
    private FrameData latestFrame;

    // The Sentis inference engine that executes the YOLO model
    private Worker worker;

    // The expected input dimensions read from the model (width, height)
    private Vector2Int inputSize;

    // Event fired after each inference run with all detections above the confidence threshold
    public event Action<
        List<(Texture roi, float confidence, FrameData frame)>
    > onDetectionsReady;


    public event Action<Texture, float, FrameData> sendYOLOROI;

    // Internal list of detections found in the current inference run
    private List<(Rect boundingBox, float confidence, FrameData frame)> detections =
        new List<(Rect, float, FrameData)>();

    private void Awake()
    {
        // Load the YOLO model from the assigned ModelAsset
        var model = ModelLoader.Load(modelAsset);

        // Read expected input dimensions from the model: shape is (1, 3, H, W)
        inputSize = new Vector2Int(model.inputs[0].shape.Get(2), model.inputs[0].shape.Get(3));

        // Create the inference worker using CPU backend
        worker = new Worker(model, BackendType.CPU);
    }

    private void OnEnable()
    {
        // Subscribe to the tensor event from CameraTextureToTensor
        cameraTextureToTensor.sendTensor += onNewTensor;
    }

    private void OnDisable()
    {
        // Unsubscribe to prevent memory leaks
        cameraTextureToTensor.sendTensor -= onNewTensor;
    }

    /// Stores the latest tensor and its associated FrameData for the next inference run
    private void onNewTensor(Tensor<float> tensor, FrameData frame)
    {
        latestTensor = tensor;
        latestFrame = frame;
    }

    /// Continuously runs inference in a coroutine loop
    private IEnumerator Start()
    {
        while (true)
        {
            yield return runInference();
        }
    }

    /// Runs one inference pass: schedules the model, reads output async, and parses detections
    private IEnumerator runInference()
    {
        // Wait until we have received at least one tensor from CameraTextureToTensor
        if (latestTensor == null)
        {
            yield return null;
            yield break;
        }

        // Snapshot the FrameData so it stays consistent even if a new tensor arrives during inference
        FrameData frame = latestFrame;

        // Feed the tensor into the YOLO model
        PipelineProfiler.begin("YOLO Inference");
        worker.Schedule(latestTensor);

        // Start async readback of the output tensor to avoid blocking the main thread
        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>)
            .ReadbackAndCloneAsync()
            .GetAwaiter();

        // Yield each frame until the async readback is complete
        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        // Retrieve the output tensor (shape: 1, 5, 8400)
        PipelineProfiler.end("YOLO Inference");
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
        PipelineProfiler.begin("YOLO Parse");
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

        PipelineProfiler.end("YOLO Parse");
        PipelineProfiler.set("Detections", detections.Count);

        // Log results for debugging (visible via ADB logcat or Meta Quest Developer Hub)
        if (detections.Count > 0)
        {
            Debug.Log(
                $"[Sentis] Detected {detections.Count} ads. Top confidence: {detections[0].confidence:0.00}"
            );

            // Temporarily only uses the first detection to filter out dupes

            var firstDetection = detections[0];

            Rect box = firstDetection.boundingBox;
            Texture cropTexture = firstDetection.frame.currentTexture;

            Rect pixelBox = new Rect(
            box.x * cropTexture.width,
            box.y * cropTexture.height,
            box.width * cropTexture.width,
            box.height * cropTexture.height
            );

            Texture croppedROI = TextureCropper.CropBoundingBox(pixelBox,cropTexture);


            /*

            if (lastCropped != null)
            {
            viewCroppedImage.Show(lastCropped);
            Debug.Log("We have a detected ad!");                
            } else
            {
            Debug.Log("crop snea");
            }
            */

            sendYOLOROI?.Invoke(croppedROI,firstDetection.confidence,firstDetection.frame);
            Debug.Log("hej jag skickade event");

        }
        else
        {
            Debug.Log("[Sentis] No ads detected this frame.");
        }

        /*
        // Notify subscribers that new detections are available
        if (detections.Count > 0)
        {
            onDetectionsReady?.Invoke(detections);
        }
        */

        yield return null;
    }

    private void OnDestroy()
    {
        // Release the inference worker to free native memory
        worker?.Dispose();
    }
}
