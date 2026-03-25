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
using System.Linq;
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

    [Header("Debug")]
    [SerializeField]
    private bool debugLogSceneStatus = true;

    [SerializeField]
    private float debugSceneStatusLogIntervalSeconds = 2f;

    [SerializeField]
    private bool debugPlaceInFrontOnRaycastMiss = true;

    [SerializeField]
    private float debugFallbackDistanceMeters = 1.5f;

    // Holds active blocks per TrackedObject ID
    private Dictionary<int, GameObject> activeBlocks = new Dictionary<int, GameObject>();

    // Holds spatial anchors per TrackedObject ID
    private Dictionary<int, OVRSpatialAnchor> activeSpatialAnchors =
        new Dictionary<int, OVRSpatialAnchor>();

    private float nextSceneStatusLogTime;

    private float nextSceneRaycastHintLogTime;

    private bool IsPassthroughRayReady =>
        cameraAccess != null && cameraAccess.enabled && cameraAccess.IsPlaying;

    private string GetPreferredRaycastManagerLabel()
    {
        if (Application.isEditor)
        {
            if (mockRaycastManager != null)
            {
                return "MOCK";
            }

            if (realRaycastManager != null)
            {
                return "REAL";
            }

            return "NONE";
        }

        if (realRaycastManager != null)
        {
            return "REAL";
        }

        if (mockRaycastManager != null)
        {
            return "MOCK";
        }

        return "NONE";
    }

    private bool TryGetPlacementRay(Vector2 viewportPoint, out Ray ray, out string source)
    {
        if (IsPassthroughRayReady)
        {
            ray = cameraAccess.ViewportPointToRay(viewportPoint);
            source = "PassthroughCameraAccess";
            return true;
        }

        if (Camera.main != null)
        {
            ray = Camera.main.ViewportPointToRay(new Vector3(viewportPoint.x, viewportPoint.y, 0f));
            source = "Camera.main";
            return true;
        }

        if (cameraRig != null && cameraRig.centerEyeAnchor != null)
        {
            ray = new Ray(cameraRig.centerEyeAnchor.position, cameraRig.centerEyeAnchor.forward);
            source = "OVRCameraRig.centerEyeAnchor.forward";
            return true;
        }

        ray = default;
        source = "NONE";
        return false;
    }

    private bool TryEnvironmentRaycast(Ray ray, out EnvironmentRaycastHit hit, out string manager)
    {
        // IMPORTANT: Prefer mock only in Editor. On device, prefer real raycast.
        // A very common failure mode is having a Mock manager assigned in the scene,
        // which then unintentionally gets used on Quest.
        if (Application.isEditor)
        {
            if (mockRaycastManager != null)
            {
                manager = "MOCK";
                return mockRaycastManager.Raycast(ray, out hit);
            }

            if (realRaycastManager != null)
            {
                manager = "REAL";
                return realRaycastManager.Raycast(ray, out hit);
            }
        }
        else
        {
            if (realRaycastManager != null)
            {
                manager = "REAL";
                return realRaycastManager.Raycast(ray, out hit);
            }

            if (mockRaycastManager != null)
            {
                manager = "MOCK";
                return mockRaycastManager.Raycast(ray, out hit);
            }
        }

        manager = "NONE";
        hit = default;
        return false;
    }

    private static bool HasLoadedBehaviourWithFullName(string fullName)
    {
        // Uses string matching to avoid a hard compile-time dependency on MRUK types.
        var behaviours = Resources.FindObjectsOfTypeAll<MonoBehaviour>();
        foreach (var behaviour in behaviours)
        {
            if (behaviour == null)
            {
                continue;
            }

            if (behaviour.GetType().FullName == fullName)
            {
                return true;
            }
        }

        return false;
    }

    private void LogSceneStatusIfDue()
    {
        if (!debugLogSceneStatus)
        {
            return;
        }

        if (Time.unscaledTime < nextSceneStatusLogTime)
        {
            return;
        }

        nextSceneStatusLogTime =
            Time.unscaledTime + Mathf.Max(0.1f, debugSceneStatusLogIntervalSeconds);

        bool hasMruk = HasLoadedBehaviourWithFullName("Meta.XR.MRUtilityKit.MRUK");
        bool hasOvrSceneManager = HasLoadedBehaviourWithFullName("OVRSceneManager");

        Debug.Log(
            "[BlockPlacementManager2] SceneStatus: "
                + $"time={Time.timeSinceLevelLoad:0.0}s, "
                + $"passthroughReady={IsPassthroughRayReady}, "
                + $"raycastManagers: real={(realRaycastManager != null ? (realRaycastManager.isActiveAndEnabled ? "active" : "inactive") : "null")}, "
                + $"mock={(mockRaycastManager != null ? (mockRaycastManager.isActiveAndEnabled ? "active" : "inactive") : "null")}, "
                + $"preferred={GetPreferredRaycastManagerLabel()}, "
                + $"mrukPresent={hasMruk}, ovrSceneManagerPresent={hasOvrSceneManager}"
        );
    }

    private void RemoveAnchorIfAny(int objectId)
    {
        if (
            !activeSpatialAnchors.TryGetValue(objectId, out OVRSpatialAnchor anchor)
            || anchor == null
        )
        {
            return;
        }

        anchor.Erase((anchorToErase, success) => { });
        activeSpatialAnchors.Remove(objectId);
        Destroy(anchor);
    }

    private GameObject GetOrCreateBlock(TrackedObject obj)
    {
        if (activeBlocks.TryGetValue(obj.id, out GameObject existing) && existing != null)
        {
            return existing;
        }

        GameObject block = Instantiate(blockPrefab);
        block.name = $"Block_{obj.id}";

        BlockVisualization vis = block.GetComponent<BlockVisualization>();
        if (vis != null)
        {
            vis.SetBlockData(obj.id);
        }

        if (cameraRig != null)
        {
            block.transform.SetParent(cameraRig.trackingSpace);
        }

        ApplyDebugVisibility(block);

        activeBlocks[obj.id] = block;
        return block;
    }

    private void ApplyDebugVisibility(GameObject block)
    {
        if (block == null)
        {
            return;
        }

        // Purely diagnostic: make spawned blocks hard to miss.
        var renderers = block.GetComponentsInChildren<Renderer>(true);
        foreach (var renderer in renderers)
        {
            if (renderer == null)
            {
                continue;
            }

            renderer.enabled = true;

            // renderer.material creates an instance per renderer (safe for debug).
            var materials = renderer.materials;
            for (int i = 0; i < materials.Length; i++)
            {
                Material mat = materials[i];
                if (mat == null)
                {
                    continue;
                }

                Color debugColor = new Color(1f, 0f, 1f, 1f);
                if (mat.HasProperty("_BaseColor"))
                {
                    mat.SetColor("_BaseColor", debugColor);
                }
                if (mat.HasProperty("_Color"))
                {
                    mat.SetColor("_Color", debugColor);
                }
                if (mat.HasProperty("_EmissionColor"))
                {
                    mat.EnableKeyword("_EMISSION");
                    mat.SetColor("_EmissionColor", debugColor * 2.0f);
                }
            }
        }
    }

    private void PlaceOrUpdateBlockFallbackInFront(TrackedObject obj, Ray ray, string reason)
    {
        if (!debugPlaceInFrontOnRaycastMiss)
        {
            return;
        }

        RemoveAnchorIfAny(obj.id);

        Vector3 origin = ray.origin;
        Vector3 direction = ray.direction;

        // For visibility debugging: put the fallback block straight ahead of the HMD,
        // not along the detection ray (which can point far off-center).
        if (cameraRig != null && cameraRig.centerEyeAnchor != null)
        {
            origin = cameraRig.centerEyeAnchor.position;
            direction = cameraRig.centerEyeAnchor.forward;
        }

        if (direction.sqrMagnitude > 0f)
        {
            direction = direction.normalized;
        }
        else
        {
            direction = Vector3.forward;
        }

        Vector3 position = origin + direction * Mathf.Max(0.05f, debugFallbackDistanceMeters);
        Quaternion rotation = Quaternion.LookRotation(-direction, Vector3.up);

        GameObject block = GetOrCreateBlock(obj);
        block.transform.position = position;
        block.transform.rotation = rotation;

        Debug.Log(
            $"[BlockPlacementManager2] Fallback placement for object {obj.id} ({reason}). "
                + $"pos={position}, distance={debugFallbackDistanceMeters:0.00}m"
        );
    }

    private void Awake()
    {
        Debug.Log(
            "[BlockPlacementManager2] Ready. "
                + $"PassthroughRayReady={IsPassthroughRayReady}, "
                + $"ManagersAssigned: real={(realRaycastManager != null)}, mock={(mockRaycastManager != null)}, "
                + $"Preferred={GetPreferredRaycastManagerLabel()}"
        );
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
        LogSceneStatusIfDue();

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

        Rect rect = obj.lastDetection.bboxNormalized;

        float centerX = rect.x + rect.width * 0.5f;
        float centerY = rect.y + rect.height * 0.5f;

        centerY = 1f - centerY;

        Vector2 viewportCenter = new Vector2(centerX, centerY);
        // -------------------------------------------------------
        /*
        // Perform raycast to find the correct position for the block
        Vector2 viewportCenter = new Vector2(
            obj.lastDetection.bboxNormalized.center.x,
            1f - obj.lastDetection.bboxNormalized.center.y // Invert Y for viewport coordinates
        );
        */
        if (!TryGetPlacementRay(viewportCenter, out Ray ray, out string raySource))
        {
            Debug.LogError(
                $"[BlockPlacementManager2] No valid camera source for ray. "
                    + $"cameraAccess={(cameraAccess != null)}, cameraRig={(cameraRig != null)}, Camera.main={(Camera.main != null)}"
            );
            return;
        }

        /*
        if (cameraRig != null && cameraRig.centerEyeAnchor != null)
        {
            ray = new Ray(cameraRig.centerEyeAnchor.position, cameraRig.centerEyeAnchor.forward);
            Debug.Log("TEST: Skjuter stråle rakt fram från headsetet");
        }
        else
        {
            // Fallback om cameraRig saknas (din gamla logik)
            ray = cameraAccess.ViewportPointToRay(viewportCenter);
        }
        */

        Debug.Log(
            $"[BlockPlacementManager2] Object {obj.id}: viewport={viewportCenter}, raySource={raySource}, "
                + $"ray.origin={ray.origin}, ray.direction={ray.direction}, passthroughReady={IsPassthroughRayReady}"
        );

        Debug.DrawRay(ray.origin, ray.direction * 3f, Color.magenta, 0.05f);

        bool didHit = TryEnvironmentRaycast(ray, out EnvironmentRaycastHit hit, out string manager);

        if (manager == "NONE")
        {
            Debug.LogError("[BlockPlacementManager2] No raycast manager assigned!");
            return;
        }

        if (!didHit)
        {
            PlaceOrUpdateBlockFallbackInFront(obj, ray, $"RaycastMiss manager={manager}");

            if (manager == "REAL" && Time.unscaledTime >= nextSceneRaycastHintLogTime)
            {
                nextSceneRaycastHintLogTime = Time.unscaledTime + 10f;
                Debug.LogWarning(
                    "[BlockPlacementManager2] REAL raycast is missing. "
                        + "This usually means there is no loaded Scene/Room geometry to hit. "
                        + "Check Quest Space Setup/Room scan and ensure Scene permission is granted."
                );
            }

            Debug.LogWarning(
                $"[BlockPlacementManager2] Raycast failed for object {obj.id}. "
                    + $"manager={manager}, preferred={GetPreferredRaycastManagerLabel()}, "
                    + $"passthroughReady={IsPassthroughRayReady}"
            );
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
        GameObject block = GetOrCreateBlock(obj);

        // 2. Position and orient the block based on raycast hit
        block.transform.position = hit.point;
        block.transform.rotation = Quaternion.LookRotation(hit.normal);

        // 4. Create spatial anchor for persistence if enabled
        if (useSpatialAnchors && cameraRig != null && !activeSpatialAnchors.ContainsKey(obj.id))
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
