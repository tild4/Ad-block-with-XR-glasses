/*
    ProcessOCRDetection2

    PURPOSE:
    Post-processes OCR text-detection output masks into word boxes,
    crops those word ROIs from the original frame, converts them to
    recognition tensors, and forwards them to OCR recognition.

    CURRENT FLOW:
    TextDetectionInference -> THIS -> TextRecognitionInference

    POLICY:
    - Latest batch wins.
    - The class owns incoming tensors and disposes dropped or consumed ones.
    - Processing yields during heavy CPU work to avoid monopolizing the frame.

    NOTE:
    Emitted tensor is the detected text

*/

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class ProcessOCRDetection_MVP2 : MonoBehaviour
{
    private const int MaskSize = 640;
    // private const long FrameBudgetMs = 3; // Removed: BFS now runs synchronously
    private const int MinBoxWidth = 10;
    private const int MinBoxHeight = 10;
    private const int PaddingX = 4;
    private const int PaddingY = 2;
    private const float MergeVerticalOverlap = 0.5f;
    private const float MergeHorizontalGapFactor = 2.0f;
    private const float MaxMergedAspectRatio = 6.0f;

    [SerializeField] private ViewCroppedImage viewCroppedImage;
    private RenderTexture debugPreviewRT;

    [SerializeField]
    private TextDetectionInference_MVP2 textDetectionInference;

    // Reused threshold mask for OCR text-detection output.
    private readonly bool[,] mask = new bool[MaskSize, MaskSize];
    private readonly float[,] scoreMap = new float[MaskSize, MaskSize];

    [SerializeField]
    private float maskThreshold = 0.3f;

    [SerializeField]
    private float boxScoreThreshold = 0.6f;

    [SerializeField]
    private float unclipRatio = 1.5f;

    /*
        OCR recognition model input:
        [DynamicDimension.0, 3, 48, DynamicDimension.1] in NCHW format.

        Width can vary, though multiples of 32 are recommended.
        Batch remains 1 per ROI.
    */
    [SerializeField]
    private int tensorTargetHeight = 48;

    [SerializeField]
    private int tensorTargetWidth = 320;

    /*
    [SerializeField]
    private int cropTargetHeight = 128;

    [SerializeField]
    private int cropTargetWidth = 512;
    */

    [SerializeField]
    private Material cropMaterial;

    // Reusable GPU resources
    private RenderTexture convertRenderTexture;
    private CommandBuffer commandBuffer;
    private bool isProcessing = false;

    public event Action<TextTensorsPerAd> sendCroppedROIText;

    private void Awake()
    {
        if (textDetectionInference == null || cropMaterial == null)
        {
            UnityEngine.Debug.Log("Missing asset");
            return;
        }

        convertRenderTexture = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        convertRenderTexture.Create();

        commandBuffer = new CommandBuffer();

        debugPreviewRT = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );
        debugPreviewRT.Create();
    }

    private void OnEnable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.findTextRegions += HandleNewTrackedObject;
        }
    }

    private void OnDisable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.findTextRegions -= HandleNewTrackedObject;
        }
    }

    /*
        Receives a new OCR text-detection batch.

        FLOW:
        1. Ignore empty input.
        2. Keep only the newest pending batch.
        3. Start processing if needed.
    */
    private void HandleNewTrackedObject(DetectionsPerAd advertisment)
    {
        if (advertisment.trackedObject == null || advertisment.findTextTensor == null)
        {
            return;
        }

        if (!isProcessing)
        {
            StartCoroutine(ProcessDPA(advertisment));
        }
    }

    /*
        Processes batches sequentially.

        POLICY:
        - Finish the current batch.
        - Then process the newest pending batch, if one exists.
    */
    private IEnumerator ProcessDPA(DetectionsPerAd advertisement)
    {
        // Prevents nested coroutines
        isProcessing = true;

        yield return ProcessDetectionOCR(advertisement);

        isProcessing = false;
    }

    /*
        Handles one OCR text-detection tensor.

        FLOW:
        1. Build the threshold mask.
        2. Find connected text boxes.
        3. Dispose the consumed detection tensor.
        4. Crop each text ROI and convert it to a recognition tensor.
        5. Store the frame batch for recognition.
    */
    private IEnumerator ProcessDetectionOCR(DetectionsPerAd advertisement)
    {
        Tensor<float> tensor = advertisement.findTextTensor;
        RenderTexture roiSnapshot = advertisement.roiSnapshot;
        Rect yoloBounds = advertisement.yoloBounds;
        Rect roiContentRect = advertisement.roiContentRectNormalized;

        if (tensor == null || roiSnapshot == null)
        {
            tensor?.Dispose();
            yield break;
        }

        /*
        From "heat map" tensor:
        1. Build text region bounding boxes
        2. Save all boundning boxes
        */

        PipelineProfiler.begin("OCR ProcessBFS");
        UnityEngine.Debug.Log($"[OCR] findTextTensor shape: {tensor.shape}");
        BuildMask(tensor);
        List<Rect> boundingBoxes = FindTextBoxes(mask, scoreMap);
        PipelineProfiler.end("OCR ProcessBFS");

        tensor.Dispose();

        // Merge nearby boxes on the same text line into full words/sentences
        boundingBoxes = MergeBoxesOnSameLine(boundingBoxes);

        if (boundingBoxes == null || boundingBoxes.Count == 0)
        {
            UnityEngine.Debug.Log("[OCR] No text boxes found in current ad crop.");
            viewCroppedImage?.SetDetectedWord("No text detected");
        }

        // Takes the list of bounds and crops the text regions from the frozen ROI snapshot
        List<TextTensor> croppedRois = BuildCroppedRecognitionRois(
            boundingBoxes,
            roiSnapshot,
            yoloBounds,
            roiContentRect
        );

        // Release the snapshot now that all text regions have been cropped
        roiSnapshot.Release();
        Destroy(roiSnapshot);

        TextTensorsPerAd advertisementWithTensors = new TextTensorsPerAd(
            advertisement.trackedObject,
            croppedRois
        );
        sendCroppedROIText?.Invoke(advertisementWithTensors);
    }

    private void BuildMask(Tensor<float> tensor)
    {
        var sw = Stopwatch.StartNew();
        float maxVal = 0f;
        int aboveCount = 0;

        for (int y = 0; y < MaskSize; y++)
        {
            for (int x = 0; x < MaskSize; x++)
            {
                float v = tensor[0, 0, y, x];
                if (v > maxVal) maxVal = v;
                bool above = v > maskThreshold;
                if (above) aboveCount++;
                mask[y, x] = above;
                scoreMap[y, x] = v;
            }
        }

        UnityEngine.Debug.Log($"[OCR Mask] maxVal={maxVal:F4}, aboveThreshold={aboveCount}/{MaskSize * MaskSize}, threshold={maskThreshold}, time={sw.ElapsedMilliseconds}ms");
    }

    /*
    NOTE :
    These bounding boxes are relative to the 640×640 ROI from YOLO.
    We crop directly from the frozen ROI snapshot using local coordinates.
    Full-frame coordinates are only computed for debug visualization.
    */
    private List<TextTensor> BuildCroppedRecognitionRois(
        List<Rect> boundingBoxes,
        RenderTexture roiSnapshot,
        Rect yoloBounds,
        Rect roiContentRectNormalized
    )
    {
        List<TextTensor> croppedRois = new List<TextTensor>();

        /*
        For each detected text region in the ad:
        1. Normalize coordinates for TextureCropper
        2. Crop from the frozen ROI snapshot (not the live camera frame)
        3. Convert it to a tensor
        4. Compute full-frame bounds for debug visualization
        */

        if (boundingBoxes == null || boundingBoxes.Count == 0)
        {
            return croppedRois;
        }

        foreach (Rect bounds in boundingBoxes)
        {
            UnityEngine.Debug.Log($"[OCR CropSize] Crop size: {bounds.width}x{bounds.height} pixels");
            Rect normalizedLocal = new Rect(
                bounds.x / MaskSize,
                bounds.y / MaskSize,
                bounds.width / MaskSize,
                bounds.height / MaskSize
            );

            // Crop from the frozen ROI snapshot (640×640) instead of the live camera frame
            int cropW = Mathf.Max(1, Mathf.RoundToInt(normalizedLocal.width * roiSnapshot.width));
            int cropH = Mathf.Max(1, Mathf.RoundToInt(normalizedLocal.height * roiSnapshot.height));
            RenderTexture tempCrop = RenderTexture.GetTemporary(cropW, cropH, 0, RenderTextureFormat.ARGB32);

            if (
                !TextureCropper.CropBoundingBoxTopLeft(
                    normalizedLocal,
                    roiSnapshot,
                    tempCrop,
                    cropMaterial
                )
            )
            {
                RenderTexture.ReleaseTemporary(tempCrop);
                continue;
            }

            Graphics.Blit(tempCrop, debugPreviewRT);
            viewCroppedImage?.Show(debugPreviewRT);

            // Scale uniformly into 48×320 with black padding to preserve aspect ratio
            PipelineProfiler.set("TensorContext", "OCR");
            Tensor<float> roiTensor = ConvertToTensor.convertWithAspectPad(
                tempCrop,
                convertRenderTexture,
                tensorTargetHeight,
                tensorTargetWidth,
                commandBuffer
            );

            RenderTexture.ReleaseTemporary(tempCrop);

            if (roiTensor != null)
            {
                // Full-frame coordinates for debug visualization
                Rect normalizedContentRect = ConvertModelRectToContentRect(
                    normalizedLocal,
                    roiContentRectNormalized
                );
                Rect normalizedFullFrame = ConvertLocalToFullFrameBounds(
                    normalizedContentRect,
                    yoloBounds
                );
                croppedRois.Add(new TextTensor(roiTensor, normalizedFullFrame));
            }
        }

        UnityEngine.Debug.Log($"[OCR Crop] Successfully cropped {croppedRois.Count} word regions from the ad.");

        if (croppedRois.Count == 0)
        {
            viewCroppedImage?.SetDetectedWord("No text detected");
        }

        return croppedRois;
    }

    private Rect ConvertModelRectToContentRect(Rect modelRect, Rect contentRect)
    {
        if (contentRect.width <= 1e-5f || contentRect.height <= 1e-5f)
        {
            return modelRect;
        }

        float localXMin = Mathf.InverseLerp(contentRect.xMin, contentRect.xMax, modelRect.xMin);
        float localXMax = Mathf.InverseLerp(contentRect.xMin, contentRect.xMax, modelRect.xMax);
        float localYMin = Mathf.InverseLerp(contentRect.yMin, contentRect.yMax, modelRect.yMin);
        float localYMax = Mathf.InverseLerp(contentRect.yMin, contentRect.yMax, modelRect.yMax);

        localXMin = Mathf.Clamp01(localXMin);
        localXMax = Mathf.Clamp01(localXMax);
        localYMin = Mathf.Clamp01(localYMin);
        localYMax = Mathf.Clamp01(localYMax);

        return Rect.MinMaxRect(localXMin, localYMin, localXMax, localYMax);
    }

    private Rect ConvertLocalToFullFrameBounds(Rect normalizedLocal, Rect parentYoloBounds)
    {
        return new Rect(
            parentYoloBounds.x + normalizedLocal.x * parentYoloBounds.width,
            parentYoloBounds.y + normalizedLocal.y * parentYoloBounds.height,
            normalizedLocal.width * parentYoloBounds.width,
            normalizedLocal.height * parentYoloBounds.height
        );
    }

    /*
        Merges bounding boxes that sit on the same horizontal text line.
        Two boxes merge when they overlap vertically by at least
        MergeVerticalOverlap of the shorter box AND the horizontal
        gap between them is less than MergeHorizontalGapFactor × average height.
        Repeats until no more merges occur.
    */
    private List<Rect> MergeBoxesOnSameLine(List<Rect> boxes)
    {
        if (boxes == null || boxes.Count <= 1)
        {
            return boxes;
        }

        bool merged = true;
        while (merged)
        {
            merged = false;
            for (int i = 0; i < boxes.Count; i++)
            {
                for (int j = i + 1; j < boxes.Count; j++)
                {
                    if (!ShouldMerge(boxes[i], boxes[j]))
                    {
                        continue;
                    }

                    float minX = Mathf.Min(boxes[i].xMin, boxes[j].xMin);
                    float minY = Mathf.Min(boxes[i].yMin, boxes[j].yMin);
                    float maxX = Mathf.Max(boxes[i].xMax, boxes[j].xMax);
                    float maxY = Mathf.Max(boxes[i].yMax, boxes[j].yMax);

                    float unionWidth = maxX - minX;
                    float unionHeight = maxY - minY;

                    if (unionHeight > 0 && unionWidth / unionHeight > MaxMergedAspectRatio)
                    {
                        continue;
                    }

                    boxes[i] = new Rect(minX, minY, unionWidth, unionHeight);
                    boxes.RemoveAt(j);
                    merged = true;
                    j--;
                }
            }
        }

        return boxes;
    }

    private bool ShouldMerge(Rect a, Rect b)
    {
        float overlapTop = Mathf.Max(a.yMin, b.yMin);
        float overlapBottom = Mathf.Min(a.yMax, b.yMax);
        float overlapHeight = overlapBottom - overlapTop;

        if (overlapHeight <= 0)
        {
            return false;
        }

        float shorterHeight = Mathf.Min(a.height, b.height);
        if (overlapHeight / shorterHeight < MergeVerticalOverlap)
        {
            return false;
        }

        float gap = Mathf.Max(0, Mathf.Max(a.xMin - b.xMax, b.xMin - a.xMax));
        float avgHeight = (a.height + b.height) * 0.5f;

        return gap < avgHeight * MergeHorizontalGapFactor;
    }

    /*
        Connected-component search over the threshold mask.
        Runs synchronously — profile via the log to verify timing.
    */
    private List<Rect> FindTextBoxes(bool[,] inputMask, float[,] inputScores)
    {
        var sw = Stopwatch.StartNew();

        int h = inputMask.GetLength(0);
        int w = inputMask.GetLength(1);

        bool[,] visited = new bool[h, w];
        List<Rect> boxes = new List<Rect>();

        int[] dx = { -1, 0, 1, -1, 1, -1, 0, 1 };
        int[] dy = { -1, -1, -1, 0, 0, 1, 1, 1 };

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (!inputMask[y, x] || visited[y, x])
                {
                    continue;
                }

                Queue<Vector2Int> queue = new Queue<Vector2Int>();
                queue.Enqueue(new Vector2Int(x, y));
                visited[y, x] = true;

                int minX = x;
                int maxX = x;
                int minY = y;
                int maxY = y;
                float scoreSum = inputScores[y, x];
                int pixelCount = 1;

                while (queue.Count > 0)
                {
                    Vector2Int p = queue.Dequeue();

                    for (int i = 0; i < 8; i++)
                    {
                        int nx = p.x + dx[i];
                        int ny = p.y + dy[i];

                        if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                        {
                            continue;
                        }

                        if (visited[ny, nx] || !inputMask[ny, nx])
                        {
                            continue;
                        }

                        visited[ny, nx] = true;
                        queue.Enqueue(new Vector2Int(nx, ny));
                        scoreSum += inputScores[ny, nx];
                        pixelCount++;

                        if (nx < minX)
                            minX = nx;
                        if (nx > maxX)
                            maxX = nx;
                        if (ny < minY)
                            minY = ny;
                        if (ny > maxY)
                            maxY = ny;
                    }
                }

                int width = maxX - minX + 1;
                int height = maxY - minY + 1;
                float averageScore = scoreSum / Mathf.Max(1, pixelCount);

                if (
                    width > MinBoxWidth
                    && height > MinBoxHeight
                    && averageScore >= boxScoreThreshold
                )
                {
                    boxes.Add(ExpandRect(minX, minY, maxX, maxY, w, h));
                }
            }
        }

        UnityEngine.Debug.Log($"[OCR BFS] Found {boxes.Count} text boxes, time={sw.ElapsedMilliseconds}ms");
        return boxes;
    }

    private Rect ExpandRect(int minX, int minY, int maxX, int maxY, int width, int height)
    {
        float boxWidth = maxX - minX + 1f;
        float boxHeight = maxY - minY + 1f;
        float area = boxWidth * boxHeight;
        float perimeter = Mathf.Max(1f, 2f * (boxWidth + boxHeight));
        int dynamicPad = Mathf.CeilToInt((area * Mathf.Max(0f, unclipRatio - 1f)) / perimeter);

        int padX = PaddingX + dynamicPad;
        int padY = PaddingY + dynamicPad;

        int paddedMinX = Mathf.Max(0, minX - padX);
        int paddedMinY = Mathf.Max(0, minY - padY);
        int paddedMaxX = Mathf.Min(width - 1, maxX + padX);
        int paddedMaxY = Mathf.Min(height - 1, maxY + padY);

        return new Rect(
            paddedMinX,
            paddedMinY,
            paddedMaxX - paddedMinX + 1,
            paddedMaxY - paddedMinY + 1
        );
    }

    private void OnDestroy()
    {
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

        if (debugPreviewRT != null)
        {
            debugPreviewRT.Release();
            Destroy(debugPreviewRT);
            debugPreviewRT = null;
        }
    }
}
