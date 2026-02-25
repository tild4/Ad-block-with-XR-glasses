using UnityEngine;
using Meta.XR;

public class BlockPlacementManager : MonoBehaviour
{

    [SerializeField] private PassthroughCameraAccess cameraAccess;
    [SerializeField] private EnvironmentRaycastManager raycastManager;
    [SerializeField] private OVRCameraRig cameraRig;
    [SerializeField] private GameObject blockPrefab;

    public float confidenceThreshold = 0.6f;

    public void ProcessDetection(DetectionResult result)
    {
        if (result.confidence < confidenceThreshold)
        {
            return;
        }

        Ray ray = cameraAccess.ViewportPointToRay(result.viewportPoint);

        if(raycastManager.Raycast(ray, out var hit))
        {
            GameObject block = Instantiate(blockPrefab);
            block.transform.position = hit.point;
            block.transform.rotation = Quaternion.LookRotation(hit.normal); // Orient block to surface normal

            block.transform.SetParent(cameraRig.transform); // Optional: parent to camera rig for consistent movement
        }

    }
}
