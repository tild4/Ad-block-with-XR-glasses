/// 
/// 
/// 
using System;
using Meta.XR;
using UnityEngine;

public class CaptureCameraTexture : MonoBehaviour
{
    [SerializeField] private PassthroughCameraAccess cameraAccess;          
    private Vector2 normalizedViewportPoint = new Vector2(0.5f, 0.5f);              
    public static Texture currentTexture {get; private set;}
    public static Pose currentPose {get; private set;}
    public static Ray currentRay {get; private set;}
    public static Vector2Int currentResolution {get; private set;}
    public static DateTime currentTimestamp {get; private set;}
 
    private void Update()
    {

        if(cameraAccess == null || !cameraAccess.enabled || !cameraAccess.IsPlaying || !cameraAccess.IsUpdatedThisFrame)
        {
            return;
        }
     
        currentTexture = cameraAccess.GetTexture();
        
        //PassthroughCameraAccess.CameraIntrinsics intrinsics = cameraAccess.Intrinsics;
        
        currentPose = cameraAccess.GetCameraPose();
        currentRay = cameraAccess.ViewportPointToRay(normalizedViewportPoint);
        currentResolution = cameraAccess.CurrentResolution;
        currentTimestamp = cameraAccess.Timestamp;
        
    }

    private void OnDisable()
    {
        currentTexture = null;
    }   
}
