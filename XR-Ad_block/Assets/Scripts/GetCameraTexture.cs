using System;
using Meta.XR;
using UnityEngine;

public class GetCameraTexture : MonoBehaviour
{
    [SerializeField] private PassthroughCameraAccess cameraAccess;
    private Vector2 normalizedViewportPoint = new Vector2(0.5f, 0.5f);
    public Texture currentTexture {get; private set;}
 
    // Update is called once per frame
    void Update()
    {
        if (cameraAccess.enabled)
        {
            // Texture can be accessed immediately after enabling the component.
            // The texture itself can be black for a couple of frames, but it's already safe to use.
            currentTexture = cameraAccess.GetTexture();
        }

        // Wait until PassthroughCameraAccess.IsPlaying is true
        if (cameraAccess.IsPlaying)
        {
            // Camera data is available only when IsPlaying is true
            PassthroughCameraAccess.CameraIntrinsics intrinsics = cameraAccess.Intrinsics;
            Pose pose = cameraAccess.GetCameraPose();
            Ray ray = cameraAccess.ViewportPointToRay(normalizedViewportPoint);

            // Newly added properties:
            Vector2Int resolution = cameraAccess.CurrentResolution;
            DateTime timestamp = cameraAccess.Timestamp;
        }
    }
}
