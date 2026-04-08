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

public class ProcessOCRDetection_uml : MonoBehaviour
{
    private const int MaskSize = 640;
    private const long FrameBudgetMs = 3;
    private const int MinBoxWidth = 10;
    private const int MinBoxHeight = 10;
    private const int PaddingX = 4;
    private const int PaddingY = 2;
    private const float MergeVerticalOverlap = 0.5f;
    private const float MergeHorizontalGapFactor = 2.0f;
    private const float MaxMergedAspectRatio = 6.0f;

    /*
    ============================== UI ==============================
    [SerializeField] private ViewCroppedImage viewCroppedImage;
    private RenderTexture debugPreviewRT;
    ===================================================================
    */

    [SerializeField]
    private TextDetectionInference_uml textDetectionInference;

    // Reused threshold mask for OCR text-detection output.
    private readonly bool[,] mask = new bool[MaskSize, MaskSize];

    [SerializeField]
    private float maskThreshold = 0.3f;

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
            Debug.Log("Missing asset");
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
        Rect parentYoloBounds = advertisement.trackedObject.lastDetection.bboxNormalized;
        Texture parentTexture = advertisement.trackedObject.lastDetection.frame.currentTexture;

        if (tensor == null)
        {
            yield break;
        }

        /*
        From "heat map" tensor:
        1. Build text region bounding boxes
        2. Save all boundning boxes
        */

        PipelineProfiler.begin("OCR ProcessBFS");
        Debug.Log($"[OCR] findTextTensor shape: {tensor.shape}");
        yield return BuildMaskFromTensor(tensor);

        List<Rect> boundingBoxes = null;
        yield return FindTextBoxesCoroutine(mask, result => boundingBoxes = result);
        PipelineProfiler.end("OCR ProcessBFS");

        tensor.Dispose();

        // Merge nearby boxes on the same text line into full words/sentences
        boundingBoxes = MergeBoxesOnSameLine(boundingBoxes);

        // Takes the list of bounds and crops the text regions
        List<TextTensor> croppedRois = BuildCroppedRecognitionRois(
            boundingBoxes,
            parentYoloBounds,
            parentTexture
        );

        TextTensorsPerAd advertisementWithTensors = new TextTensorsPerAd(
            advertisement.trackedObject,
            croppedRois
        );
        sendCroppedROIText?.Invoke(advertisementWithTensors);
    }

    private IEnumerator BuildMaskFromTensor(Tensor<float> tensor)
    {
        var sw = Stopwatch.StartNew();

        for (int y = 0; y < MaskSize; y++)
        {
            for (int x = 0; x < MaskSize; x++)
            {
                mask[y, x] = tensor[0, 0, y, x] > maskThreshold;
            }

            if (sw.ElapsedMilliseconds >= FrameBudgetMs)
            {
                yield return null;
                sw.Restart();
            }
        }
    }

    /*
    NOTE :
    These bounding boxes are relative to the ad region from YOLO
    Therefore the coordinates need to be converted to be relative
    to the full frame
    */
    private List<TextTensor> BuildCroppedRecognitionRois(
        List<Rect> boundingBoxes,
        Rect parentYoloBounds,
        Texture parentTexture
    )
    {
        List<TextTensor> croppedRois = new List<TextTensor>();

        /*
        For each detected text region in the ad:
        1. Normalize cooridinates for TextureCropper
        2. Convert the coordinates relative to the ad region to coordinates relative to the full frame
        3. Crop the text region
        4. Convert it to a tensor
        5. Include the relative bounds in emission for debuggning purposes / to view cropped text region
        */

        foreach (Rect bounds in boundingBoxes)
        {
            Debug.Log($"[OCR CropSize] Crop size: {bounds.width}x{bounds.height} pixels");
            Rect normalizedLocal = new Rect(
                bounds.x / MaskSize,
                bounds.y / MaskSize,
                bounds.width / MaskSize,
                bounds.height / MaskSize
            );

            Rect normalizedFullFrame = ConvertLocalToFullFrameBounds(
                normalizedLocal,
                parentYoloBounds
            );

            // Crop at natural resolution to preserve aspect ratio
            int cropW = Mathf.Max(1, Mathf.RoundToInt(normalizedFullFrame.width * parentTexture.width));
            int cropH = Mathf.Max(1, Mathf.RoundToInt(normalizedFullFrame.height * parentTexture.height));
            RenderTexture tempCrop = RenderTexture.GetTemporary(cropW, cropH, 0, RenderTextureFormat.ARGB32);

            if (!TextureCropper.CropBoundingBox(normalizedFullFrame, parentTexture, tempCrop, cropMaterial))
            {
                RenderTexture.ReleaseTemporary(tempCrop);
                continue;
            }

            /*
            ============================== UI ==============================
            Graphics.Blit(tempCrop, debugPreviewRT);
            viewCroppedImage.Show(debugPreviewRT);
            ===================================================================
            */

            // Scale uniformly into 48×320 with black padding to preserve aspect ratio
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
                croppedRois.Add(new TextTensor(roiTensor, normalizedFullFrame));
            }
        }

        Debug.Log($"[OCR Crop] Successfully cropped {croppedRois.Count} word regions from the ad.");
        return croppedRois;
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

                    // Union of both boxes
                    float minX = Mathf.Min(boxes[i].xMin, boxes[j].xMin);
                    float minY = Mathf.Min(boxes[i].yMin, boxes[j].yMin);
                    float maxX = Mathf.Max(boxes[i].xMax, boxes[j].xMax);
                    float maxY = Mathf.Max(boxes[i].yMax, boxes[j].yMax);

                    float unionWidth = maxX - minX;
                    float unionHeight = maxY - minY;

                    // Skip merge if result would be too wide for recognition
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
        // Vertical overlap check
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

        // Horizontal gap check
        float gap = Mathf.Max(0, Mathf.Max(a.xMin - b.xMax, b.xMin - a.xMax));
        float avgHeight = (a.height + b.height) * 0.5f;

        return gap < avgHeight * MergeHorizontalGapFactor;
    }

    /*
        Connected-component search over the threshold mask.
        Yields periodically so one large mask does not block the main thread.
    */
    private IEnumerator FindTextBoxesCoroutine(bool[,] inputMask, Action<List<Rect>> onComplete) 
    {
        int h = inputMask.GetLength(0);
        int w = inputMask.GetLength(1);

        bool[,] visited = new bool[h, w];
        List<Rect> boxes = new List<Rect>();

        int[] dx = { -1, 0, 1, -1, 1, -1, 0, 1 };
        int[] dy = { -1, -1, -1, 0, 0, 1, 1, 1 };

        var sw = Stopwatch.StartNew();

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

                        if (nx < minX)
                            minX = nx;
                        if (nx > maxX)
                            maxX = nx;
                        if (ny < minY)
                            minY = ny;
                        if (ny > maxY)
                            maxY = ny;
                    }

                    if (sw.ElapsedMilliseconds >= FrameBudgetMs)
                    {
                        yield return null;
                        sw.Restart();
                    }
                }

                int width = maxX - minX + 1;
                int height = maxY - minY + 1;

                if (width > MinBoxWidth && height > MinBoxHeight)
                {
                    int paddedMinX = Mathf.Max(0, minX - PaddingX);
                    int paddedMinY = Mathf.Max(0, minY - PaddingY);
                    int paddedMaxX = Mathf.Min(w - 1, maxX + PaddingX);
                    int paddedMaxY = Mathf.Min(h - 1, maxY + PaddingY);

                    boxes.Add(
                        new Rect(
                            paddedMinX,
                            paddedMinY,
                            paddedMaxX - paddedMinX + 1,
                            paddedMaxY - paddedMinY + 1
                        )
                    );
                }
            }

            if (sw.ElapsedMilliseconds >= FrameBudgetMs)
            {
                yield return null;
                sw.Restart();
            }
        }

        onComplete?.Invoke(boxes);
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
