using UnityEngine;
using Meta.XR;


/// <summary>
/// Responible for converting 2D detection results (from YOLO) 
/// into 3D world-space blocker placement using Meta XR tools.
/// </summary>
public class BlockPlacementManager : MonoBehaviour
{
    // Reference to Meta's passthrough camera system.
    // Used to convert normalized viewport coordinates to 3D ray.
    [SerializeField] private PassthroughCameraAccess cameraAccess;

    // Utilitykits Raycastmanager.
    // Raycasts against real-world spatial mesh.
    [SerializeField] private EnvironmentRaycastManager raycastManager;

    // Used to spawned blockers can be place correctly in the world and move with the user.
    [SerializeField] private OVRCameraRig cameraRig;

    // The object to be spawned, if we want to use a customprefab later, we can just swap it out here / editor.
    [SerializeField] private GameObject blockPrefab;

    // Detection confidence thresehold to spawn a blocker, can be adjusted.
    public float confidenceThreshold = 0.6f;
    private GameObject currentBlock;


    // Take in detection result from YOLO, chekc confidence and raycast blocker into world.
    public void ProcessDetection(DetectionResult result)
    {
        // Not confident enough, skip placing a block.
        if (result.confidence < confidenceThreshold)
        {
            return;
        }

        // Convert normalized viewport coordinates (0-1) to a ray in world space.
        // YOLO gives us the center of the detected object in normalized viewport space.
        // Accounts for: Camera projection, field of view, current head pose
        // Here we can probably replace by our own methods if we want future optimizations.
        // Will require some linear algebra to convert 2D screen point to a 3D ray based on camera intrinsics and head pose.
        Ray ray = cameraAccess.ViewportPointToRay(result.viewportPoint);



        if(raycastManager.Raycast(ray, out var hit))
        {

            //Spawn only one block and adjust it's placement, instead of spawning multiple blocks for each detection.
            if (currentBlock == null)
            {
                currentBlock = Instantiate(blockPrefab);
                currentBlock.transform.SetParent(cameraRig.trackingSpace);
            }

            currentBlock.transform.position = hit.point;
            currentBlock.transform.rotation = Quaternion.LookRotation(hit.normal); 

            currentBlock.transform.SetParent(cameraRig.trackingSpace); 
        }

    }
}
