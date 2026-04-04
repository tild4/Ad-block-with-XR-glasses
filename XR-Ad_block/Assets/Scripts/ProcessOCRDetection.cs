using Unity.InferenceEngine;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;
using System.Diagnostics;
using UnityEngine.Rendering;

public class ProcessOCRDetection : MonoBehaviour
{
    [SerializeField] TextDetectionInference textDetectionInference;

    // Should be the same as targetWidth and targetHeight
    private bool[,] mask = new bool[640, 640];
    [SerializeField] private float threshold = 0.3f; 

    [SerializeField] private int roiBatchThreshold = 5;

    /*
        Exact tensor input settings for ONNX model is:
        [DynamicDimension.0, 3, 48, DynamicDimension.1]
        in NCHW format.

        Therefore tensorTargetWidth can vary, though multiples of 32
        are recommended. Batch is still 1 per ROI.
    */

    [SerializeField]
    private int tensorTargetHeight = 48;

    [SerializeField]
    private int tensorTargetWidth = 320;

    // Reused GPU resources for conversion
    private RenderTexture convertRenderTexture;

    private RenderTexture croppedROI;

    private CommandBuffer commandBuffer;

    private List<(List<Tensor<float>>, FrameData)> processedROIBatch = new List<(List<Tensor<float>>, FrameData)>();

    public event Action<List<(List<Tensor<float>>, FrameData)>> sendCroppedROIText;


    private void Awake()
    {
        //Allocate reusable GPU resources

        convertRenderTexture = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        croppedROI = new RenderTexture(
            tensorTargetWidth,
            tensorTargetHeight,
            0,
            RenderTextureFormat.ARGB32
        );

        croppedROI.Create();

        convertRenderTexture.Create();

        commandBuffer = new CommandBuffer();
    }

    private void OnEnable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.decodeDetectionTensors += decodeDetections;           
        }
    }

    private void OnDisable()
    {
        if (textDetectionInference != null)
        {
            textDetectionInference.decodeDetectionTensors -= decodeDetections;           
        }
    }

    private void decodeDetections(List<(Tensor<float>, FrameData, Rect)> detections)
    {        
        if (detections == null || detections.Count == 0)
        {
            return;
        }

        foreach(var detection in detections)
        {
            Tensor<float> tensor = detection.Item1;
            FrameData frame = detection.Item2;

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

            List<Tensor<float>> croppedROIs = new List<Tensor<float>>();

            foreach (var bounds in boundingBoxes) {
                
                /*
                if (!TextureCropper.CropBoundingBox(bounds,frame.currentTexture,croppedROI))
                {
                    continue;
                }
                */

                Tensor<float> roiTensor = ConvertToTensor.convert(croppedROI,convertRenderTexture,tensorTargetHeight,tensorTargetWidth,commandBuffer);

                if (roiTensor != null)
                {
                    croppedROIs.Add(roiTensor);
                }
            }
            

            processedROIBatch.Add((croppedROIs,frame));
        }

        if(processedROIBatch.Count > 0)
        {
        var sendBatch = new List<(List<Tensor<float>>, FrameData)>(processedROIBatch);
        sendCroppedROIText?.Invoke(sendBatch);
        processedROIBatch.Clear();
        UnityEngine.Debug.Log("hellllo send to recognition pls");
        }

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

private void OnDestroy()
{
    // Dispose any ROI tensors still waiting in processedROIBatch
    if (processedROIBatch != null)
    {
        foreach (var frameBatch in processedROIBatch)
        {
            List<Tensor<float>> roiTensors = frameBatch.Item1;

            if (roiTensors == null)
            {
                continue;
            }

            foreach (Tensor<float> tensor in roiTensors)
            {
                tensor?.Dispose();
            }

            roiTensors.Clear();
        }

        processedROIBatch.Clear();
    }

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

    if (croppedROI != null)
    {
        croppedROI.Release();
        Destroy(croppedROI);
        croppedROI = null;
    }
}


}
