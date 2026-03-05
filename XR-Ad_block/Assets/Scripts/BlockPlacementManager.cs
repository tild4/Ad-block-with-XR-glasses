using Meta.XR;
using UnityEngine;

/// <summary>
/// Responible for converting 2D detection results (from YOLO)
/// into 3D world-space blocker placement using Meta XR tools.
/// </summary>
public class BlockPlacementManager : MonoBehaviour
{
    // Reference to Meta's passthrough camera system.
    // Used to convert normalized viewport coordinates to 3D ray.
    [SerializeField]
    private PassthroughCameraAccess cameraAccess;

    // Utilitykits Raycastmanager.
    // Raycasts against real-world spatial mesh.
    [SerializeField]
    private EnvironmentRaycastManager raycastManager;

    // Used so spawned blockers can be place correctly in the world and move with the user.
    [SerializeField]
    private OVRCameraRig cameraRig;

    // The object to be spawned, if we want to use a customprefab later, we can just swap it out here / editor.
    [SerializeField]
    private GameObject blockPrefab;

    // Detection confidence thresehold to spawn a blocker, can be adjusted.
    public float confidenceThreshold = 0.6f;
    private float lastDetectionTime;
    public float persistenceDuration = 0.5f; // How long the block should persist after the last detection
    private GameObject currentBlock;

    void Update()
    {
    if (currentBlock != null)
        {
            if (Time.time - lastDetectionTime > persistenceDuration)
            {
                currentBlock.SetActive(false);
            }
            else
            {
                currentBlock.SetActive(true);
            }
        }
    }

    // Take in detection result from YOLO, chekc confidence and raycast blocker into world.
    public void ProcessDetection(Rect rect, float confidence, FrameData frame)
    {
        Debug.Log($"Confidence: {confidence} | Rect: {rect}");
        // Not confident enough, skip placing a block.
        if (confidence < confidenceThreshold)
        {
            return;
        }

        float centerX = rect.x + rect.width * 0.5f;
        float centerY = rect.y + rect.height * 0.5f;

        centerY = 1f - centerY;

        Vector2 viewportPoint = new Vector2(centerX, centerY);
        Debug.Log("Viewport point: " + viewportPoint);

        // Convert normalized viewport coordinates (0-1) to a ray in world space.
        // Accounts for: Camera projection, field of view, current head pose
        // Here we can probably replace by our own methods if we want future optimizations.
        // Will require some linear algebra to convert 2D screen point to a 3D ray based on camera intrinsics and head pose.

        Ray ray = cameraAccess.ViewportPointToRay(viewportPoint);



        /*
        //Debugging method
        Transform centerEye = cameraRig.centerEyeAnchor;

        Ray ray = new Ray(
            centerEye.position,
            centerEye.forward
        );
        */

        Debug.DrawRay(ray.origin, ray.direction * 5f, Color.red, 1f);
        Debug.Log("Ray origin: " + ray.origin);
        Debug.Log("Ray direction: " + ray.direction);


        if (raycastManager.Raycast(ray, out var hit))
        {
            lastDetectionTime = Time.time;
            Debug.Log("RAYCAST HIT AT: " + hit.point);
            Vector3 targetPosition = hit.point + hit.normal * 0.01f;

            //Spawn only one block and adjust it's placement, instead of spawning multiple blocks for each detection.
            if (currentBlock == null)
            {
                currentBlock = Instantiate(blockPrefab);
                currentBlock.transform.position = targetPosition;
            }
            else
            {
                currentBlock.transform.position = Vector3.Lerp(
                    currentBlock.transform.position,
                    targetPosition,
                    Time.deltaTime * 10f); // Smoothly move the block to the new position
            }

            Debug.Log("Block position: " + currentBlock.transform.position);

            if (hit.normal.sqrMagnitude > 0.0001f) // Avoid zero-length normals
            {
                currentBlock.transform.rotation = Quaternion.LookRotation(hit.normal);
            }

            //currentBlock.transform.SetParent(cameraRig.trackingSpace);
        }
        else
        {
            Debug.Log("RAYCAST MISSED");
        }
        
    }
}
