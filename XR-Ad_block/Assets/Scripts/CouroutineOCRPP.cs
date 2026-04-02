using Unity.InferenceEngine;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

public class ProcessOCRDetection : MonoBehaviour
{
    [SerializeField] private TextDetectionInference textDetectionInference;

    // Should match OCR text detection output resolution
    private readonly bool[,] mask = new bool[640, 640];

    [SerializeField] private float threshold = 0.3f;

    /*
        Latest batch waiting to be processed.

        POLICY:
        - Only one batch is actively processed at a time
        - If a newer batch arrives while one is pending, the older pending batch is dropped
        - This keeps the AR pipeline fresh instead of building stale latency
    */
    private List<(Tensor<float>, FrameData)> pendingBatch;

    private bool isProcessing = false;

    /*
        Accumulates cropped ROI output for the currently processed batch.
        This is copied before being emitted through the event.
    */
    private readonly List<(List<Texture>, FrameData)> processedROIBatch = new List<(List<Texture>, FrameData)>();

    public event Action<List<(List<Texture>, FrameData)>> sendCroppedROIText;

    private void OnEnable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.decodeDetectionTensors += OnNewDetectionBatch;
        }
    }

    private void OnDisable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.decodeDetectionTensors -= OnNewDetectionBatch;
        }
    }

    /*
        Event callback should stay lightweight.

        RESPONSIBILITIES:
        - Ignore empty input
        - Replace any older pending batch with the newest one
        - Dispose tensors from overwritten pending work
        - Start coroutine if not already running

        OWNERSHIP:
        - This class owns and disposes tensors that it receives
        - If a pending batch is overwritten before processing starts,
          this class must dispose those dropped tensors too
    */
    private void OnNewDetectionBatch(List<(Tensor<float>, FrameData)> detections)
    {
        if (detections == null || detections.Count == 0)
        {
            return;
        }

        // Copy outer list so we do not depend on producer's internal buffer
        var latestBatch = new List<(Tensor<float>, FrameData)>(detections);

        // If an older batch was waiting but never processed, drop it safely
        if (pendingBatch != null)
        {
            DisposeTensorBatch(pendingBatch);
        }

        pendingBatch = latestBatch;

        if (!isProcessing)
        {
            StartCoroutine(ProcessQueue());
        }
    }

    /*
        Processes batches sequentially.

        POLICY:
        - Finish current batch
        - Then process the newest pending batch, if any
        - Older pending batches are overwritten before they ever start

        YIELDING:
        - Yields between detections
        - Also yields periodically during heavy loops inside ProcessOneDetection
    */
    private IEnumerator ProcessQueue()
    {
        isProcessing = true;

        while (pendingBatch != null)
        {
            var batch = pendingBatch;
            pendingBatch = null;

            processedROIBatch.Clear();

            foreach (var detection in batch)
            {
                yield return ProcessOneDetection(detection);

                // Give control back between detections
                yield return null;
            }

            if (processedROIBatch.Count > 0)
            {
                var sendBatch = new List<(List<Texture>, FrameData)>(processedROIBatch);
                sendCroppedROIText?.Invoke(sendBatch);
                processedROIBatch.Clear();

                Debug.Log("hellllo send to recognition pls");
            }
        }

        isProcessing = false;
    }

    /*
        Process one OCR text-detection tensor.

        FLOW:
        1. Build bool mask from tensor
        2. Find connected components / text boxes
        3. Dispose tensor after CPU processing is done
        4. Build ROI list for this frame
        5. Add result to batch buffer

        OWNERSHIP:
        - Tensor is consumed and disposed here
    */
    private IEnumerator ProcessOneDetection((Tensor<float>, FrameData) detection)
    {
        Tensor<float> tensor = detection.Item1;
        FrameData frame = detection.Item2;

        if (tensor == null)
        {
            yield break;
        }

        PipelineProfiler.begin("OCR ProcessBFS");

        // Build threshold mask
        for (int y = 0; y < 640; y++)
        {
            for (int x = 0; x < 640; x++)
            {
                mask[y, x] = tensor[0, 0, y, x] > threshold;
            }

            // Periodically yield during long scan
            if (y % 32 == 0)
            {
                yield return null;
            }
        }

        List<Rect> boundingBoxes = null;
        yield return FindTextBoxesCoroutine(mask, result => boundingBoxes = result);

        PipelineProfiler.end("OCR ProcessBFS");

        tensor.Dispose();

        var croppedROIs = new List<Texture>();

        /*
            Future crop step goes here.

            Example:
            foreach (var bounds in boundingBoxes)
            {
                var roi = cropCode(bounds, ...);
                if (roi != null)
                {
                    croppedROIs.Add(roi);
                }
            }
        */

        processedROIBatch.Add((croppedROIs, frame));
    }

    /*
        Coroutine version of connected-component search.

        Yields periodically so one large tensor does not monopolize the main thread.
    */
    private IEnumerator FindTextBoxesCoroutine(bool[,] inputMask, Action<List<Rect>> onComplete)
    {
        int h = inputMask.GetLength(0);
        int w = inputMask.GetLength(1);

        bool[,] visited = new bool[h, w];
        List<Rect> boxes = new List<Rect>();

        int[] dx = { -1, 1, 0, 0 };
        int[] dy = { 0, 0, -1, 1 };

        int workCounter = 0;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (!inputMask[y, x] || visited[y, x])
                    continue;

                Queue<Vector2Int> queue = new Queue<Vector2Int>();
                queue.Enqueue(new Vector2Int(x, y));
                visited[y, x] = true;

                int minX = x, maxX = x;
                int minY = y, maxY = y;

                while (queue.Count > 0)
                {
                    var p = queue.Dequeue();
                    workCounter++;

                    for (int i = 0; i < 4; i++)
                    {
                        int nx = p.x + dx[i];
                        int ny = p.y + dy[i];

                        if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                            continue;

                        if (visited[ny, nx] || !inputMask[ny, nx])
                            continue;

                        visited[ny, nx] = true;
                        queue.Enqueue(new Vector2Int(nx, ny));

                        if (nx < minX) minX = nx;
                        if (nx > maxX) maxX = nx;
                        if (ny < minY) minY = ny;
                        if (ny > maxY) maxY = ny;
                    }

                    // Periodically yield during BFS expansion
                    if (workCounter % 5000 == 0)
                    {
                        yield return null;
                    }
                }

                // Inclusive bounds -> +1 so single-pixel width/height stays valid
                int width = maxX - minX + 1;
                int height = maxY - minY + 1;

                if (width > 10 && height > 10)
                {
                    boxes.Add(new Rect(minX, minY, width, height));
                }
            }

            // Periodically yield during full image scan too
            if (y % 32 == 0)
            {
                yield return null;
            }
        }

        onComplete?.Invoke(boxes);
    }

    /*
        Dispose tensors in a batch that is being dropped before processing.

        This is necessary because this class owns the tensors once it accepts them.
    */
    private void DisposeTensorBatch(List<(Tensor<float>, FrameData)> batch)
    {
        if (batch == null)
        {
            return;
        }

        foreach (var item in batch)
        {
            item.Item1?.Dispose();
        }
    }

    private void OnDestroy()
    {
        // If something is still pending when object is destroyed, dispose it
        DisposeTensorBatch(pendingBatch);
        pendingBatch = null;
    }
}