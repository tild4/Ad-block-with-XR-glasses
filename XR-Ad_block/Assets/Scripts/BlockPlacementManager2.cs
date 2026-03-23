/*
    BlockPlacementManager2

    PURPOSE:
    Handles the physical spawning, positioning, and persistence of 3D blocker
    prefabs in the AR environment based on tracked AI detections.

    ARCHITECTURE:
    - Event-Driven: Subscribes to TrackingManager.onTrackedObjectsUpdated.
    - Dual-Raycast System: Automatically switches between MockEnvironmentRaycastManager
      (for Editor testing) and realRaycastManager (for Meta Quest hardware).
    - Spatial Persistence: Uses Meta's OVRSpatialAnchor to lock blocks to
      real-world physical locations.
    - Lifecycle Management:
        1. CreateBlockWithAnchor: Spawns prefab and initializes visualization.
        2. UpdateBlock: Moves block if the detection moves.
        3. RemoveBlock: Cleans up GameObjects and erases Spatial Anchors from memory.

    IMPORTANT:
    Requires a 'BlockVisualization' component on the blockPrefab to show IDs.
    Parenting to cameraRig.trackingSpace ensures blocks stay relative to the user's room.
*/

using System.Collections.Generic;
using Meta.XR;
using UnityEngine;

public class BlockPlacementManager2 : MonoBehaviour
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
    private TrackingManager trackingManager;

    [Header("Prefab")]
    [SerializeField]
    private GameObject blockPrefab;

    [Header("Settings")]
    [SerializeField]
    private bool useSpatialAnchors = true;

    // Holds active blocks per TrackedObject ID
    private Dictionary<int, GameObject> activeBlocks = new Dictionary<int, GameObject>();

    // Holds spatial anchors per TrackedObject ID
    private Dictionary<int, OVRSpatialAnchor> activeSpatialAnchors =
        new Dictionary<int, OVRSpatialAnchor>();

    /*
        For logging and debugging: Verifies that all dependencies are assigned in the Inspector.
    */
    private void Start()
    {
        Debug.Log($"=== BlockPlacementManager2 Setup ===");
        Debug.Log($"mockRaycastManager: {(mockRaycastManager != null ? "✓" : "✗ NULL")}");
        Debug.Log($"realRaycastManager: {(realRaycastManager != null ? "✓" : "✗ NULL")}");
        Debug.Log($"trackingManager: {(trackingManager != null ? "✓" : "✗ NULL")}");
        Debug.Log($"blockPrefab: {(blockPrefab != null ? "✓" : "✗ NULL")}");
    }

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
    }

    /*
        Handles the logic for placing a new block or updating an existing one based on the tracked object's state.
        - If the object should not be blocked and a block exists, it removes the block.
        - If the object should be blocked, it performs a raycast to find the correct position in the environment and either creates a new block or updates the existing one.
    */
    private void PlaceOrUpdateBlock(TrackedObject obj)
    {
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

        // Perform raycast to find the correct position for the block
        Vector2 viewportCenter = new Vector2(
            obj.lastDetection.bboxNormalized.center.x,
            1f - obj.lastDetection.bboxNormalized.center.y // Invert Y for viewport coordinates
        );

        Ray ray;
        if (cameraAccess != null)
        {
            ray = cameraAccess.ViewportPointToRay(viewportCenter);
        }
        else
        {
            if (Camera.main == null)
            {
                Debug.LogError("No Camera.main found!");
                return;
            }
            ray = Camera.main.ViewportPointToRay(
                new Vector3(viewportCenter.x, viewportCenter.y, 0)
            );
        }

        EnvironmentRaycastHit hit;
        bool didHit = false;

        if (mockRaycastManager != null)
        {
            didHit = mockRaycastManager.Raycast(ray, out hit);
        }
        else if (realRaycastManager != null)
        {
            didHit = realRaycastManager.Raycast(ray, out hit);
        }
        else
        {
            Debug.LogError("No raycast manager assigned!");
            return;
        }

        if (!didHit)
        {
            Debug.LogWarning($"Raycast failed for object {obj.id}");
            return;
        }

        if (!activeBlocks.ContainsKey(obj.id))
        {
            CreateBlockWithAnchor(obj, hit);
        }
        else
        {
            UpdateBlock(obj, hit);
        }
    }

    /*
        Creates a new block GameObject at the specified hit location and sets up a spatial anchor for it.
        - Instantiates the block prefab and names it based on the tracked object's ID.
        - Sets the block's position and rotation based on the raycast hit.
        - Parents the block to the camera rig's tracking space to maintain relative positioning in the room.
        - If spatial anchors are enabled, it creates an OVRSpatialAnchor component, saves it, and stores it in the activeSpatialAnchors dictionary for later management.
    */
    private void CreateBlockWithAnchor(TrackedObject obj, EnvironmentRaycastHit hit)
    {
        // 1. Instantiate block prefab
        GameObject block = Instantiate(blockPrefab);
        block.name = $"Block_{obj.id}";

        // Set block visualization (e.g., show ID) if the component exists
        BlockVisualization vis = block.GetComponent<BlockVisualization>();
        if (vis != null)
        {
            vis.SetBlockData(obj.id);
        }

        // 2. Position and orient the block based on raycast hit
        block.transform.position = hit.point;
        block.transform.rotation = Quaternion.LookRotation(hit.normal);

        // 3. Parent to camera rig's tracking space for room-relative positioning
        if (cameraRig != null)
        {
            block.transform.SetParent(cameraRig.trackingSpace);
        }

        // 4. Create spatial anchor for persistence if enabled
        if (useSpatialAnchors && cameraRig != null)
        {
            OVRSpatialAnchor spatialAnchor = block.AddComponent<OVRSpatialAnchor>();
            spatialAnchor.Save(
                (anchor, success) =>
                {
                    if (success)
                    {
                        Debug.Log($"✓ Spatial anchor saved for object {obj.id}");
                    }
                    else
                    {
                        Debug.LogWarning($"✗ Failed to save spatial anchor for object {obj.id}");
                    }
                }
            );
            // Store the spatial anchor reference for later cleanup
            activeSpatialAnchors[obj.id] = spatialAnchor;
        }
        // 5. Store the block reference for later updates/removal
        activeBlocks[obj.id] = block;
        Debug.Log($"Created block for object {obj.id}");
    }

    /*
        Updates the position and rotation of an existing block based on the new raycast hit information.
        - Retrieves the block GameObject from the activeBlocks dictionary using the tracked object's ID.
        - Updates the block's position to the new hit point and rotates it to align with the hit normal.
    */
    private void UpdateBlock(TrackedObject obj, EnvironmentRaycastHit hit)
    {
        GameObject block = activeBlocks[obj.id];
        // Update position and rotation based on new raycast hit
        block.transform.position = hit.point;
        block.transform.rotation = Quaternion.LookRotation(hit.normal);
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
        List<int> allIds = new List<int>(activeBlocks.Keys);
        foreach (var id in allIds)
        {
            RemoveBlock(id);
        }
    }
}
