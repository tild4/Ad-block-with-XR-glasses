using UnityEngine;
using System;
using System.Collections.Generic;

public class CropDetections : MonoBehaviour
{
    /*
    public SentisInferenceManager inferenceManager;

    public event Action<RenderTexture> OnCropReady;

    private void OnEnable()
    {
        //subscribe to detections from the inference manager
        inferenceManager.onDetectionsReady += OnDetection;
    }

    private void OnDisable()
    {
        inferenceManager.onDetectionsReady -= OnDetection;
    }

    private void OnDetection(List<(Rect boundingBox, float confidence, FrameData frame)> detections)
    {
        foreach (var detection in detections)
        {
            HandleDetection(detection.boundingBox, detection.frame);
        }
    }

    private void HandleDetection(Rect boundingBox, FrameData frame)
    {
        Texture source = frame.currentTexture;
        RenderTexture cropped = TextureCropper.CropBoundingBox(boundingBox, source);

        OnCropReady?.Invoke(cropped);
    }
    */
}