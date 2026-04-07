using System.Collections;
using System.Collections.Generic;
using System.Linq;
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

    // Reference to XR camera rig. Used if spatial alignment with headset pose is required.
    [SerializeField]
    private OVRCameraRig cameraRig;

    // Reference to SentisInferenceManager to subscribe to detection events.
    [SerializeField]
    private SentisInferenceManager inferenceManager;

    // Prefab spawned to visually block ads
    [SerializeField]
    private GameObject blockPrefab;

    // Detection confidence thresehold to spawn a blocker, can be adjusted.
    public float confidenceThreshold = 0.6f;
    private float lastDetectionTime;
    public float persistenceDuration = 0.5f; // How long the block should persist after the last detection
    private GameObject currentBlock;

    // In Update, we check if the current block should still be visible based on the last detection time.
    // If the detection is older than the persistence duration, we hide the block until a new detection comes in.
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

    void Awake()
    {
        if (inferenceManager == null)
        {
            inferenceManager = FindObjectOfType<SentisInferenceManager>();
        }

        if (inferenceManager == null)
        {
            Debug.LogError("NO SentisInferenceManager FOUND IN SCENE");
            return;
        }

        Debug.Log("Found Sentis: " + inferenceManager.GetInstanceID());

        inferenceManager.onDetectionsReady += HandleDetections;
    }

    void OnDisable()
    {
        // Unsubscribe to prevent memory leaks
        inferenceManager.onDetectionsReady -= HandleDetections;
    }

    // Handle the list of detections from SentisInferenceManager, pick the best one, and process it.
    private void HandleDetections(
        List<(Rect boundingBox, float confidence, FrameData frame)> detections
    )
    {
        if (detections.Count == 0)
        {
            return;
        }

        var best = detections.OrderByDescending(d => d.confidence).First();
        ProcessDetection(best.boundingBox, best.confidence, best.frame);
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

        // Convert normalized rect (0-1) to viewport coordinates (0-1) with center point.
        float centerX = rect.x + rect.width * 0.5f;
        float centerY = rect.y + rect.height * 0.5f;

        // Invert Y coordinate because YOLO coordinates are at the top-left of the image,
        // while Unity's viewport coordinates are at the bottom-left.
        centerY = 1f - centerY;

        Vector2 viewportPoint = new Vector2(centerX, centerY);
        Debug.Log("Viewport point: " + viewportPoint);

        // Convert the viewport coordinates to a ray in world space using the passthrough camera projection.
        Ray ray = cameraAccess.ViewportPointToRay(viewportPoint);

        /*
        //Debugging method
        Transform centerEye = cameraRig.centerEyeAnchor;

        Ray ray = new Ray(
            centerEye.position,
            centerEye.forward
        );
        */

        // Raycast return true if it hits real-world geometry (like walls, furniture) and provides hit info (point, normal)
        if (raycastManager.Raycast(ray, out var hit))
        {
            // Update the last detection time to keep the block visible
            lastDetectionTime = Time.time;
            Debug.Log("RAYCAST HIT AT: " + hit.point);

            // Offset the block slightly along the normal to prevent z-fighting with the surface
            Vector3 targetPosition = hit.point + hit.normal * 0.01f;

            //Spawn only one block and adjust it's placement, instead of spawning multiple blocks for each detection.
            if (currentBlock == null)
            {
                currentBlock = Instantiate(blockPrefab);
                currentBlock.transform.position = targetPosition;
            }
            else
            {
                // Smoothly move the existing block to the new position
                currentBlock.transform.position = Vector3.Lerp(
                    currentBlock.transform.position,
                    targetPosition,
                    Time.deltaTime * 10f
                ); // Can be adjusted for faster/slower movement
            }

            Debug.Log("Block position: " + currentBlock.transform.position);

            if (hit.normal.sqrMagnitude > 0.0001f) // Avoid zero-length normals
            {
                currentBlock.transform.rotation = Quaternion.LookRotation(hit.normal);
            }
        }
        else
        {
            Debug.Log("RAYCAST MISSED");
        }
    }
}
