using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

public class BlockPlacementManager_MVP2 : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField]
    private PassthroughCameraAccess cameraAccess;

    // Two separate fields - use mock for Editor, real for Quest
    [SerializeField]
    private EnvironmentRaycastManager realRaycastManager;

    [SerializeField]
    private MockEnvironmentRaycastManager mockRaycastManager;

    [SerializeField]
    private OVRCameraRig cameraRig;

    [SerializeField]
    private TrackingManager_MVP2 trackingManager;

    [Header("Prefab")]
    [SerializeField]
    private GameObject blockPrefab;

    [Header("Settings")]
    [SerializeField]
    private bool useSpatialAnchors = true;

    [SerializeField]
    private bool useCameraPlanePlacement = true;

    [SerializeField]
    private float placementPlaneOffsetMeters = 0.01f;

    [SerializeField]
    private bool logRaycastMisses;

    // Holds active blocks per TrackedObject ID
    private readonly Dictionary<int, GameObject> activeBlocks = new Dictionary<int, GameObject>();

    // Holds spatial anchors per TrackedObject ID
    private readonly Dictionary<int, OVRSpatialAnchor> activeSpatialAnchors =
        new Dictionary<int, OVRSpatialAnchor>();

    /*
        Subscribes to tracking updates when enabled.
    */
    private void OnEnable()
    {

        if (trackingManager != null)
        {
            trackingManager.onTrackedObjectsUpdated += SyncBlocksWithTracking;
        }
    }

    /*
        Unsubscribes from tracking updates when disabled.
    */
    private void OnDisable()
    {
        if (trackingManager != null)
        {
            trackingManager.onTrackedObjectsUpdated -= SyncBlocksWithTracking;
        }
    }

    /*
        Core method that synchronizes the active blocks with the current tracked objects.
        - For each active track, it calls PlaceOrUpdateBlock to ensure a block exists and is correctly positioned.
        - It also checks for any blocks that no longer have a corresponding tracked object and removes them.
    */
    private void SyncBlocksWithTracking(List<TrackedObject> activeTracks)
    {
        // --- START PROFILING ---
        PipelineProfiler.begin("5. Block Placement (3D)");

        //Update or create blocks for all currently tracked objects
        foreach (var obj in activeTracks)
        {
            PlaceOrUpdateBlock(obj);
        }

        //Remove blocks that no longer have a corresponding tracked object
        List<int> idsToRemove = new List<int>();
        foreach (var id in activeBlocks.Keys)
        {
            bool stillTracked = activeTracks.Exists(obj => obj.id == id);
            if (!stillTracked)
            {
                idsToRemove.Add(id);
            }
        }

        foreach (var id in idsToRemove)
        {
            RemoveBlock(id);
        }

        PipelineProfiler.set("ActiveBlocks", activeBlocks.Count);
        // --- END PROFILING ---
        PipelineProfiler.end("5. Block Placement (3D)");
    }

    private static Rect ToViewportRect(Rect yoloNormalizedRect)
    {
        // YOLO: top-left origin. Viewport: bottom-left origin.
        float viewportYMin = 1f - yoloNormalizedRect.yMax;
        return new Rect(
            yoloNormalizedRect.xMin,
            viewportYMin,
            yoloNormalizedRect.width,
            yoloNormalizedRect.height
        );
    }

    private bool TryRaycastEnvironment(Ray ray, out EnvironmentRaycastHit hit)
    {
        if (Application.isEditor)
        {
            if (mockRaycastManager != null)
            {
                return mockRaycastManager.Raycast(ray, out hit);
            }

            if (realRaycastManager != null)
            {
                return realRaycastManager.Raycast(ray, out hit);
            }
        }
        else
        {
            if (realRaycastManager != null)
            {
                return realRaycastManager.Raycast(ray, out hit);
            }

            if (mockRaycastManager != null)
            {
                return mockRaycastManager.Raycast(ray, out hit);
            }
        }

        hit = default;
        return false;
    }

    private bool TryComputeCameraPlanePlacement(
        Rect yoloNormalizedRect,
        Pose cameraPose,
        Vector3 depthPoint,
        out Vector3 worldPosition,
        out Quaternion worldRotation
    )
    {
        worldPosition = default;
        worldRotation = default;

        if (cameraAccess == null || !cameraAccess.enabled || !cameraAccess.IsPlaying)
        {
            return false;
        }

        float distance = Vector3.Distance(cameraPose.position, depthPoint);
        if (distance <= 0.0001f)
        {
            return false;
        }

        Rect viewportRect = ToViewportRect(yoloNormalizedRect);
        Ray centerRay = cameraAccess.ViewportPointToRay(viewportRect.center, cameraPose);
        worldPosition = centerRay.GetPoint(distance);

        // Vector from camera to the placement point.
        Vector3 fromCamera = (worldPosition - cameraPose.position);
        if (fromCamera.sqrMagnitude < 1e-6f)
        {
            return false;
        }

        // We want the block (and its ID label) to face the camera.
        Vector3 towardCamera = (-fromCamera).normalized;

        worldRotation = Quaternion.LookRotation(towardCamera, Vector3.up);
        // Nudge slightly toward the camera to avoid embedding into the hit surface.
        worldPosition += towardCamera * placementPlaneOffsetMeters;
        return true;
    }

    /*
     * Helper method to compute the world size of the block based on the YOLO bounding box and the camera's field of view at the given depth.
    */
    private Vector2 ComputeWorldSize(Rect yoloRect, Pose cameraPose, float depth)
    {
        float halfFovRad = Mathf.Deg2Rad * (Camera.main.fieldOfView / 2f);
        float viewportWidthAtDepth = 2f * depth * Mathf.Tan(halfFovRad);
        float viewportHeightAtDepth = viewportWidthAtDepth * ((float)cameraAccess.CurrentResolution.y / cameraAccess.CurrentResolution.x);

        float padding = 0.7f; // Optional padding to make blocks slightly smaller than the bounding box
        return new Vector2(
            yoloRect.width * viewportWidthAtDepth * padding,
            yoloRect.height * viewportHeightAtDepth * padding
        );
    }

    /*
        Handles the logic for placing a new block or updating an existing one based on the tracked object's state.
        - If the object should not be blocked and a block exists, it removes the block.
        - If the object should be blocked, it performs a raycast to find the correct position in the environment and either creates a new block or updates the existing one.
    */
    private void PlaceOrUpdateBlock(TrackedObject obj)
    {
        try
        {
            Debug.Log($"[Block] Attempting placement for object {obj.id}, shouldBlock={obj.shouldBlock}");
            // If the object should not be blocked but we have an active block, remove it
            if (!obj.shouldBlock && activeBlocks.ContainsKey(obj.id))
            {
                RemoveBlock(obj.id);
                return;
            }

            // If the object should not be blocked and we don't have an active block, do nothing
            if (!obj.shouldBlock)
            {
                return;
            }

            if (Camera.main == null && (cameraAccess == null || !cameraAccess.enabled))
            {
                Debug.LogError("No camera available for ViewportPointToRay.");
                return;
            }

            Rect yoloRect = obj.lastDetection.bboxNormalized;
            Pose cameraPose = obj.lastDetection.frame.currentPose;
            Rect viewportRect = ToViewportRect(yoloRect);

            bool usingPassthroughRay =
                cameraAccess != null && cameraAccess.enabled && cameraAccess.IsPlaying;

            Ray ray = usingPassthroughRay
                ? cameraAccess.ViewportPointToRay(viewportRect.center, cameraPose)
                : Camera.main.ViewportPointToRay(
                    new Vector3(viewportRect.center.x, viewportRect.center.y, 0f)
                );

            if (!TryRaycastEnvironment(ray, out EnvironmentRaycastHit hit))
            {
                if (logRaycastMisses)
                {
                    Debug.LogWarning($"Raycast failed for object {obj.id}");
                }
                return;
            }

            Vector3 position;
            Quaternion rotation;

            if (
                useCameraPlanePlacement
                && TryComputeCameraPlanePlacement(
                    yoloRect,
                    cameraPose,
                    hit.point,
                    out position,
                    out rotation
                )
            )
            {
                // Use plane placement.
            }
            else
            {

                position = hit.point;
                Vector3 towardCamera = (usingPassthroughRay
                    ? cameraPose.position
                    : Camera.main.transform.position) - position;
                if (towardCamera.sqrMagnitude < 1e-6f)
                {
                    // Ray points from camera -> world; invert to get world -> camera.
                    towardCamera = -ray.direction;
                }
                rotation = Quaternion.LookRotation(towardCamera.normalized, Vector3.up);
            }

            float depth = Vector3.Distance(cameraPose.position, hit.point);
            Vector2 worldSize = ComputeWorldSize(yoloRect, cameraPose, depth);

            if (!activeBlocks.ContainsKey(obj.id))
            {
                CreateBlockWithAnchor(obj, position, rotation, worldSize);
            }   
            else
            {
                UpdateBlock(obj, position, rotation, worldSize);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[Block] Exception in PlaceOrUpdateBlock: {e.Message}\n{e.StackTrace}");
        }
    }

    /*
        Creates a new block GameObject at the specified hit location and sets up a spatial anchor for it.
        - Instantiates the block prefab and names it based on the tracked object's ID.
        - Sets the block's position and rotation based on the raycast hit.
        - Parents the block to the camera rig's tracking space to maintain relative positioning in the room.
        - If spatial anchors are enabled, it creates an OVRSpatialAnchor component, saves it, and stores it in the activeSpatialAnchors dictionary for later management.
    */
    private void CreateBlockWithAnchor(TrackedObject obj, Vector3 position, Quaternion rotation, Vector2 size)
    {
        GameObject block = Instantiate(blockPrefab);
        block.name = $"Block_{obj.id}";
        Vector3 worldScale = new Vector3(size.x, size.y, 0.01f);


        BlockVisualization vis = block.GetComponent<BlockVisualization>();
        if (vis != null)
        {
            vis.SetBlockData(obj.id);
            vis.SetTargetTransform(position, rotation, worldScale);
        }
        else
        {
            block.transform.position = position;
            block.transform.rotation = rotation;
            block.transform.localScale = worldScale;
        }

        if (cameraRig != null)
        {
            block.transform.SetParent(cameraRig.trackingSpace);
        }

        // 4. Create spatial anchor for persistence if enabled
        if (useSpatialAnchors && cameraRig != null && !activeSpatialAnchors.ContainsKey(obj.id))
        {
            OVRSpatialAnchor spatialAnchor = block.AddComponent<OVRSpatialAnchor>();
            spatialAnchor.Save(
                (anchor, success) =>
                {
                    if (!success)
                    {
                        Debug.LogWarning($"Failed to save spatial anchor for object {obj.id}");
                    }
                }
            );
            // Store the spatial anchor reference for later cleanup
            activeSpatialAnchors[obj.id] = spatialAnchor;
        }

        activeBlocks[obj.id] = block;
    }

    /*
        Updates the position and rotation of an existing block based on the new raycast hit information.
        - Retrieves the block GameObject from the activeBlocks dictionary using the tracked object's ID.
        - Updates the block's position to the new hit point and rotates it to align with the hit normal.
    */
    private void UpdateBlock(TrackedObject obj, Vector3 position, Quaternion rotation, Vector2 size)
    {
        GameObject block = activeBlocks[obj.id];
        block.transform.position = position;
        block.transform.rotation = rotation;

        Vector3 worldScale = new Vector3(size.x, size.y, 0.01f);
        BlockVisualization vis = block.GetComponent<BlockVisualization>();
        if (vis != null)
        {
            vis.SetTargetTransform(position, rotation, worldScale);
        }
        else
        {
            block.transform.position = position;
            block.transform.rotation = rotation;
            block.transform.localScale = worldScale;
        }
    }

    /*
        Removes a block GameObject and its associated spatial anchor (if it exists) based on the tracked object's ID.
        - Checks if the block exists in the activeBlocks dictionary; if not, it returns early.
        - If a spatial anchor exists for this block, it calls Erase to remove it from the environment and then removes the reference from the activeSpatialAnchors dictionary.
        - Destroys the block GameObject and removes its reference from the activeBlocks dictionary.
    */
    private void RemoveBlock(int objectId)
    {
        if (!activeBlocks.ContainsKey(objectId))
        {
            return;
        }

        if (activeSpatialAnchors.ContainsKey(objectId))
        {
            OVRSpatialAnchor anchor = activeSpatialAnchors[objectId];
            anchor.Erase((anchorToErase, success) => { });
            activeSpatialAnchors.Remove(objectId);
        }

        Destroy(activeBlocks[objectId]);
        activeBlocks.Remove(objectId);
    }

    /*
        Ensures that all active blocks and their spatial anchors are cleaned up when the manager is destroyed.
        - Iterates through all active block IDs and calls RemoveBlock to clean up each one.
    */
    private void OnDestroy()
    {
        var allIds = new List<int>(activeBlocks.Keys);
        foreach (int id in allIds)
        {
            RemoveBlock(id);
        }
    }
    /* 
        Public method that can be called by others to clear all blocks and their associated spatial anchors.
        Used by AppStateManager when transitioning back to the StartScreen state to ensure a clean slate.
    */
    public void ClearAllBlocks()
    {
        var allIds = new List<int>(activeBlocks.Keys);
        foreach (int id in allIds)
        {
            RemoveBlock(id);
        }
    }
}
