using UnityEngine;
using UnityEngine.Android;

public class PermissionManager : MonoBehaviour
{
    private const string HeadsetCameraPermission =
        "horizonos.permission.HEADSET_CAMERA";

    void Start()
    {
        getCameraPermission();
    }

    void getCameraPermission()
    {
        if (!Permission.HasUserAuthorizedPermission(HeadsetCameraPermission))
        {
            Permission.RequestUserPermission(HeadsetCameraPermission);
        }
    }
}