/*
    CaptureCameraFrame

    PURPOSE:
    Reads camera data from Meta Quest passthrough camera
    Packages relevant data into a FrameData struct
    Emits it through an event (newFrame)

    ARCHITECTURE:
    - Polls hardware state in Update()
    - If valid new frame exists → creates FrameData
    - Broadcasts it via event

    IMPORTANT:
    This class does NOT process images.
    It only collects and distributes camera frame data.
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

    // Used by ViewportPointToRay indicates the center of input camera
    private Vector2 normalizedViewportPoint = new Vector2(0.5f, 0.5f);

    private float lastProcessTime = 0f;

    /*
    public struct FrameData
    {
        public Texture currentTexture;
        public Pose currentPose;
        public Ray currentRay;
        public Vector2Int currentResolution;
        public DateTime currentTimestamp;
    }
    */

    // Event-based architecture
    public event Action<FrameData> newFrame;

    // Update every frame
    private void Update()
    {
        if (Time.time - lastProcessTime < processingInterval)
        {
            return;
        }

        // --- START PROFILING ---
        PipelineProfiler.begin("1. Capture Camera Data");

        lastProcessTime = Time.time;

        // Guard rail
        if (cameraAccess == null || !cameraAccess.enabled || !cameraAccess.IsPlaying)
        {
            Debug.Log("failed frame");
            PipelineProfiler.end("1. Capture Camera Data");
            return;
        }

        //PassthroughCameraAccess.CameraIntrinsics intrinsics = cameraAccess.Intrinsics; *vet ej om detta behövs*

        // Contruct frame

        FrameData frame = new FrameData
        {
            currentTexture = cameraAccess.GetTexture(),
            currentPose = cameraAccess.GetCameraPose(),
            currentRay = cameraAccess.ViewportPointToRay(normalizedViewportPoint),
            currentResolution = cameraAccess.CurrentResolution,
            currentTimestamp = cameraAccess.Timestamp,
        };

        PipelineProfiler.set("Cam Res", $"{frame.currentResolution.x}x{frame.currentResolution.y}");

        // Invokes event -> new frame ready to be utilized
        newFrame?.Invoke(frame);

        // --- END PROFILING ---
        PipelineProfiler.end("1. Capture Camera Data");

        Debug.Log("frame invoked");
    }
}
