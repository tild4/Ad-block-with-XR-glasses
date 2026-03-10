using System;
using Meta.XR;
using UnityEngine;

public class CaptureSimulatorFrame : MonoBehaviour
{
    [SerializeField]
    private Camera simulatorCamera;

    [SerializeField]
    private RenderTexture renderTexture;

    // Event-based architecture
    public event Action<FrameData> newFrame;

    // Update every frame
    private void Update()
    {
        FrameData frame = new FrameData
        {
            currentTexture = renderTexture,
            currentPose = new Pose(
                simulatorCamera.transform.position,
                simulatorCamera.transform.rotation
            ),
            currentRay = new Ray(
                simulatorCamera.transform.position,
                simulatorCamera.transform.forward
            ),
            currentResolution = new Vector2Int(renderTexture.width, renderTexture.height),
            currentTimestamp = DateTime.Now,
        };
        newFrame?.Invoke(frame);
    }
}
