/*
    Summary:
    Converts OCR text-detection heatmaps into cropped text ROI tensors for
    text recognition.

    Pipeline:
    TextDetectionInference -> ProcessOCRDetection -> TextRecognitionInference
*/

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class ProcessOCRDetection : MonoBehaviour
{
    private const int MaskSize = 640;

    private const int MinBoxWidth = 10;
    private const int MinBoxHeight = 10;
    private const int PaddingX = 4;
    private const int PaddingY = 2;
    private const float MergeVerticalOverlap = 0.5f;
    private const float MergeHorizontalGapFactor = 2.0f;
    private const float MaxMergedAspectRatio = 6.0f;

    [SerializeField]
    private TextDetectionInference textDetectionInference;

    private readonly bool[,] mask = new bool[MaskSize, MaskSize];
    private readonly float[,] scoreMap = new float[MaskSize, MaskSize];

    [SerializeField]
    private float maskThreshold = 0.3f;

    [SerializeField]
    private float boxScoreThreshold = 0.6f;

    [SerializeField]
    private float unclipRatio = 1.5f;

    [SerializeField]
    private Material cropMaterial;

    private const int TensorTargetHeight = 48;
    private const int TensorTargetWidth = 320;
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
            TensorTargetWidth,
            TensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        convertRenderTexture.Create();

        commandBuffer = new CommandBuffer();
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

    private void HandleNewTrackedObject(DetectionsPerAd advertisement)
    {
        if (advertisement.trackedObject == null || advertisement.findTextTensor == null)
        {
            UnityEngine.Debug.Log(
                $"[ProcessOCR] Early Exit for ID {advertisement.trackedObject?.id}. Skip OCR."
            );
            return;
        }

        if (!isProcessing)
        {
            StartCoroutine(ProcessDPA(advertisement));
        }
    }

    private IEnumerator ProcessDPA(DetectionsPerAd advertisement)
    {
        isProcessing = true;

        yield return ProcessDetectionOCR(advertisement);

        isProcessing = false;
    }

    private IEnumerator ProcessDetectionOCR(DetectionsPerAd advertisement)
    {
        Tensor<float> tensor = advertisement.findTextTensor;
        RenderTexture roiSnapshot = advertisement.roiSnapshot;

        if (tensor == null || roiSnapshot == null)
        {
            tensor?.Dispose();
            yield break;
        }

        PipelineProfiler.begin("OCR ProcessBFS");
        UnityEngine.Debug.Log($"[OCR] findTextTensor shape: {tensor.shape}");
        BuildMask(tensor);
        List<Rect> boundingBoxes = FindTextBoxes(mask, scoreMap);
        PipelineProfiler.end("OCR ProcessBFS");

        tensor.Dispose();

        boundingBoxes = MergeBoxesOnSameLine(boundingBoxes);

        if (boundingBoxes == null || boundingBoxes.Count == 0)
        {
            UnityEngine.Debug.Log("[OCR] No text boxes found in current ad crop.");
        }

        List<TextTensor> croppedRois = BuildCroppedRecognitionRois(boundingBoxes, roiSnapshot);

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
                if (v > maxVal)
                    maxVal = v;
                bool above = v > maskThreshold;
                if (above)
                    aboveCount++;
                mask[y, x] = above;
                scoreMap[y, x] = v;
            }
        }

        UnityEngine.Debug.Log(
            $"[OCR Mask] maxVal={maxVal:F4}, aboveThreshold={aboveCount}/{MaskSize * MaskSize}, threshold={maskThreshold}, time={sw.ElapsedMilliseconds}ms"
        );
    }

    private List<TextTensor> BuildCroppedRecognitionRois(
        List<Rect> boundingBoxes,
        RenderTexture roiSnapshot
    )
    {
        List<TextTensor> croppedRois = new List<TextTensor>();

        if (boundingBoxes == null || boundingBoxes.Count == 0)
        {
            return croppedRois;
        }

        foreach (Rect bounds in boundingBoxes)
        {
            UnityEngine.Debug.Log(
                $"[OCR CropSize] Crop size: {bounds.width}x{bounds.height} pixels"
            );
            Rect normalizedLocal = new Rect(
                bounds.x / MaskSize,
                bounds.y / MaskSize,
                bounds.width / MaskSize,
                bounds.height / MaskSize
            );

            int cropW = Mathf.Max(1, Mathf.RoundToInt(normalizedLocal.width * roiSnapshot.width));
            int cropH = Mathf.Max(1, Mathf.RoundToInt(normalizedLocal.height * roiSnapshot.height));
            RenderTexture tempCrop = RenderTexture.GetTemporary(
                cropW,
                cropH,
                0,
                RenderTextureFormat.ARGB32
            );

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

            PipelineProfiler.set("TensorContext", "OCR");
            Tensor<float> roiTensor = ConvertToTensor.convertWithAspectPad(
                tempCrop,
                convertRenderTexture,
                TensorTargetHeight,
                TensorTargetWidth,
                commandBuffer
            );

            RenderTexture.ReleaseTemporary(tempCrop);

            if (roiTensor != null)
            {
                croppedRois.Add(new TextTensor(roiTensor));
            }
        }

        UnityEngine.Debug.Log(
            $"[OCR Crop] Successfully cropped {croppedRois.Count} word regions from the ad."
        );

        return croppedRois;
    }

    /*
        Merges bounding boxes that sit on the same horizontal text line.
        Two boxes merge when they overlap vertically by at least
        MergeVerticalOverlap of the shorter box AND the horizontal
        gap between them is less than MergeHorizontalGapFactor times average height.
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

        UnityEngine.Debug.Log(
            $"[OCR BFS] Found {boxes.Count} text boxes, time={sw.ElapsedMilliseconds}ms"
        );
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
    }
}
