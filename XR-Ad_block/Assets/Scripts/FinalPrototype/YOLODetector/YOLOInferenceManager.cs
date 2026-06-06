/*
    Summary:
    Runs YOLO object detection on camera frames and emits normalized ad
    bounding boxes with the capture-time FrameData.

    Pipeline:
    CaptureCameraFrame -> YOLOInferenceManager -> YOLOPostProcessor
*/
using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class YOLOInferenceManager : MonoBehaviour
{
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField, Range(0f, 1f)]
    private float confidenceThreshold = 0.5f;

    [SerializeField]
    private CaptureCameraFrame captureCameraFrame;

    private Tensor<float> latestTensor;
    private FrameData latestFrame;
    private Worker worker;
    private Vector2Int inputSize;
    private RenderTexture renderTexture;
    private CommandBuffer commandBuffer;

    public event Action<
        List<(Rect boundingBox, float confidence, FrameData frame)>
    > onDetectionsReady;

    private List<(Rect boundingBox, float confidence, FrameData frame)> detections =
        new List<(Rect, float, FrameData)>();

    private void Awake()
    {
        if (modelAsset == null || captureCameraFrame == null)
        {
            Debug.Log("missing asset");
            return;
        }

        var model = ModelLoader.Load(modelAsset);

        // Model input shape is (1, 3, H, W).
        inputSize = new Vector2Int(model.inputs[0].shape.Get(2), model.inputs[0].shape.Get(3));

        renderTexture = new RenderTexture(inputSize.y, inputSize.x, 0, RenderTextureFormat.ARGB32);
        renderTexture.Create();

        commandBuffer = new CommandBuffer();
        worker = new Worker(model, BackendType.CPU);
    }

    private void OnEnable()
    {
        if (captureCameraFrame != null)
        {
            captureCameraFrame.newFrame += onNewFrame;
        }
    }

    private void OnDisable()
    {
        if (captureCameraFrame != null)
        {
            captureCameraFrame.newFrame -= onNewFrame;
        }
    }

    private void onNewFrame(FrameData frame)
    {
        if (latestTensor != null)
        {
            return;
        }

        PipelineProfiler.set("TensorContext", "YOLO");
        latestTensor = ConvertToTensor.convert(
            frame.currentTexture,
            renderTexture,
            inputSize.x,
            inputSize.y,
            commandBuffer
        );

        latestFrame = frame;
    }

    private IEnumerator Start()
    {
        while (true)
        {
            yield return runInference();
        }
    }

    private IEnumerator runInference()
    {
        if (latestTensor == null)
        {
            yield return null;
            yield break;
        }

        FrameData frame = latestFrame;
        Tensor<float> inputTensor = latestTensor;
        latestTensor = null;

        PipelineProfiler.begin("3. AI Inference");
        PipelineProfiler.set("Model Input", $"{inputSize.x}x{inputSize.y}");
        worker.Schedule(inputTensor);

        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>)
            .ReadbackAndCloneAsync()
            .GetAwaiter();

        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        PipelineProfiler.end("3. AI Inference");

        inputTensor.Dispose();

        using var output = outputAwaiter.GetResult();

        if (output == null)
        {
            yield break;
        }

        // Output shape is (1, 5, anchors): [centerX, centerY, width, height, confidence].
        PipelineProfiler.begin("4. Post-Processing (Boxes)");
        detections.Clear();

        int numDetections = output.shape[2];

        for (int i = 0; i < numDetections; i++)
        {
            float confidence = output[0, 4, i];

            if (confidence < confidenceThreshold)
            {
                continue;
            }

            float centerX = output[0, 0, i];
            float centerY = output[0, 1, i];
            float width = output[0, 2, i];
            float height = output[0, 3, i];

            float x1 = (centerX - width / 2f) / inputSize.x;
            float y1 = (centerY - height / 2f) / inputSize.y;
            float normalizedWidth = width / inputSize.x;
            float normalizedHeight = height / inputSize.y;

            detections.Add(
                (new Rect(x1, y1, normalizedWidth, normalizedHeight), confidence, frame)
            );
        }

        PipelineProfiler.end("4. Post-Processing (Boxes)");
        PipelineProfiler.set("Detections", detections.Count);

        if (detections.Count > 0)
        {
            Debug.Log(
                $"[YOLO] Detected {detections.Count} ads. Top confidence: {detections[0].confidence:0.00}"
            );

            var sendDetections = new List<(Rect boundingBox, float confidence, FrameData frame)>(
                detections
            );
            onDetectionsReady?.Invoke(sendDetections);
        }
        else
        {
            Debug.Log("[YOLO] No ads detected this frame.");
        }

        yield return null;
    }

    private void OnDestroy()
    {
        latestTensor?.Dispose();
        worker?.Dispose();

        if (commandBuffer != null)
        {
            commandBuffer.Release();
            commandBuffer = null;
        }

        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
            renderTexture = null;
        }
    }
}
