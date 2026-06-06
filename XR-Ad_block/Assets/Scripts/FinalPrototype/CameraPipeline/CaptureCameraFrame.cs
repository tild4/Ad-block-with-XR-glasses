/*
    Summary:
    Samples the Meta Quest passthrough camera at a fixed interval and emits
    camera texture data for the detection pipeline.

    Pipeline:
    PassthroughCameraAccess -> CaptureCameraFrame -> YOLOInferenceManager

    Note:
    This project uses and adapts sample code provided through the Meta XR SDK.

    Copyright © Meta Platform Technologies, LLC and its affiliates.
    All rights reserved.
*/
using System;
using Meta.XR;
using UnityEngine;

public class CaptureCameraFrame : MonoBehaviour
{
    [SerializeField]
    private PassthroughCameraAccess cameraAccess;

    [SerializeField]
    private float processingInterval = 0.15f;

    private float lastProcessTime = 0f;

    public event Action<FrameData> newFrame;

    private void Update()
    {
        if (Time.time - lastProcessTime < processingInterval)
        {
            return;
        }

        lastProcessTime = Time.time;

        if (cameraAccess == null || !cameraAccess.enabled || !cameraAccess.IsPlaying)
        {
            Debug.Log("failed frame");
            return;
        }

        PipelineProfiler.begin("1. Capture Camera Data");

        FrameData frame = new FrameData
        {
            currentTexture = cameraAccess.GetTexture(),
            currentPose = cameraAccess.GetCameraPose(),
            currentResolution = cameraAccess.CurrentResolution,
        };

        PipelineProfiler.set("Cam Res", $"{frame.currentResolution.x}x{frame.currentResolution.y}");

        PipelineProfiler.end("1. Capture Camera Data");

        newFrame?.Invoke(frame);
    }
}
