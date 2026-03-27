using Unity.InferenceEngine;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

public class ProcessOCRDetection : MonoBehaviour
{
    [SerializeField] TextDetectionInference textDetectionInference;

    // Should be the same as targetWidth and targetHeight
    private bool[,] mask = new bool[640, 640];
    [SerializeField] private float threshold = 0.3f; 

    [SerializeField] private int roiBatchThreshold = 5;

    private int droppedROI = 0;

    public event Action<List<Texture2D>, FrameData> sendCroppedROIText;

    private void OnEnable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.decodeDetectionTensor += decodeDetection;           
        }
    }

    private void OnDisable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.decodeDetectionTensor -= decodeDetection;           
        }
    }

    private void decodeDetection(Tensor<float> tensor, FrameData frame)
    {
        
        if (tensor == null)
        {
            return;
        }

        PipelineProfiler.begin("OCR ProcessBFS");
        for (int y = 0; y < 640; y++)
        {
            for (int x = 0; x < 640; x++)
            {
                mask[y, x] = tensor[0, 0, y, x] > threshold;
            }
        }

        List<Rect> boundingBoxes = findTextBoxes(mask);
        PipelineProfiler.end("OCR ProcessBFS");

        tensor.Dispose();

        List<Texture2D> croppedROI = new List<Texture2D>();

        PipelineProfiler.begin("Crop roi batch");
        foreach (Rect box in boundingBoxes)
        {

            Rect normalizedBox = new Rect(
            box.x * frame.currentResolution.x / 640f, 
            box.y * frame.currentResolution.y / 640f,
            box.width * frame.currentResolution.x / 640f,
            box.height * frame.currentResolution.y / 640f
            );

            Texture2D cropped = TextureCropper.CropBoundingBox(normalizedBox, frame.currentTexture);

            if (cropped != null && croppedROI.Count < roiBatchThreshold)
            {
                croppedROI.Add(cropped);
                Debug.Log("added cropped frame yes");
            } else if (cropped == null)
            {
                Debug.Log("null cropped frame");
            } else
            {
                droppedROI++;
            }
        }
        PipelineProfiler.end("Crop roi batch");
        Debug.Log("Number of dropped roi : " + droppedROI);
        droppedROI = 0;

        sendCroppedROIText?.Invoke(croppedROI, frame);
        Debug.Log("hellllo send to recognition pls");

    }

    private List<Rect> findTextBoxes(bool[,] mask)
    {
        int h = mask.GetLength(0);
        int w = mask.GetLength(1);

        bool[,] visited = new bool[h, w];
        List<Rect> boxes = new List<Rect>();

        int[] dx = { -1, 1, 0, 0 };
        int[] dy = { 0, 0, -1, 1 };

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (!mask[y, x] || visited[y, x])
                    continue;

                // Start BFS
                Queue<Vector2Int> queue = new Queue<Vector2Int>();
                queue.Enqueue(new Vector2Int(x, y));
                visited[y, x] = true;

                int minX = x, maxX = x;
                int minY = y, maxY = y;

                while (queue.Count > 0)
                {
                    var p = queue.Dequeue();

                    for (int i = 0; i < 4; i++)
                    {
                        int nx = p.x + dx[i];
                        int ny = p.y + dy[i];

                        if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                            continue;

                        if (visited[ny, nx] || !mask[ny, nx])
                            continue;

                        visited[ny, nx] = true;
                        queue.Enqueue(new Vector2Int(nx, ny));

                        // Expand bounds
                        if (nx < minX) minX = nx;
                        if (nx > maxX) maxX = nx;
                        if (ny < minY) minY = ny;
                        if (ny > maxY) maxY = ny;
                    }
                }

                // Filter tiny noise blobs
                int width = maxX - minX;
                int height = maxY - minY;

                if (width > 10 && height > 10)
                {
                    boxes.Add(new Rect(minX, minY, width, height));
                }
            }
        }

        return boxes;
    }



}
