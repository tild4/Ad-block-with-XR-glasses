using UnityEngine;
using System;
using System.Collections.Generic;

public class CropDetections : MonoBehaviour
{
    /*
    public SentisInferenceManager inferenceManager;

    public event Action<Texture2D> OnCropReady;

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
        //convert bounding box from normalized to pixel coordinates
        Rect boundingBoxPixel = new Rect(
            boundingBox.x * frame.currentResolution.x,
            boundingBox.y * frame.currentResolution.y,
            boundingBox.width * frame.currentResolution.x,
            boundingBox.height * frame.currentResolution.y
        );
        Texture2D cropped = TextureCropper.CropBoundingBox(boundingBoxPixel, source);

        OnCropReady?.Invoke(cropped);
    }
    */
}