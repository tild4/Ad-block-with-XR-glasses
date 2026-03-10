using UnityEngine;

//This is used to synchronize the movement of the main camera and the frame capture camera when using the simulator
public class CameraSync : MonoBehaviour
{
    [SerializeField]
    private Transform mainCamera; // CenterEyeAnchor

    private void LateUpdate()
    {
        // Copy position and rotation every frame
        transform.position = mainCamera.position;
        transform.rotation = mainCamera.rotation;
    }
}
