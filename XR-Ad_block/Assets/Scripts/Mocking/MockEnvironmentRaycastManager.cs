/*
MockEnvironmentRaycastManager

PURPOSE:
Simulates real-world spatial mapping by providing "fake" raycast hits
in the Unity Editor. This allows the BlockPlacementManager2 to function
without Meta's EnvironmentRaycastManager (which requires a physical room scan).

ARCHITECTURE:
- Geometric Simulation: Calculates a hit point by projecting a distance
  along the ray's origin and direction.
- Normal Calculation: Automatically sets the surface normal to be inverse
  of the ray's direction, ensuring that placed blocks face the camera.
- Interface Compatibility: Designed to match the signature of Meta's
  official raycast system to allow seamless switching between Mock and Real
  managers.

IMPORTANT:
Adjust 'hitDistance' to simulate how far away the "virtual wall" should be
during testing.
*/

using Meta.XR;
using UnityEngine;

public class MockEnvironmentRaycastManager : MonoBehaviour
{
    [Header("Mock Settings")]
    [SerializeField]
    private float hitDistance = 2f; // Distance from the ray origin to the simulated hit point

    [SerializeField]
    private bool alwaysHit = true; // The raycast will always return a hit. Set to false to simulate misses.

    /*
        Simulates a raycast by calculating a hit point at a specified distance along the ray.
    */
    public bool Raycast(Ray ray, out EnvironmentRaycastHit hit)
    {
        if (alwaysHit)
        {
            hit = new EnvironmentRaycastHit
            {
                point = ray.origin + ray.direction * hitDistance, // Calculate hit point at a fixed distance along the ray
                normal = -ray.direction, // Set normal to face the ray origin, simulating a flat surface facing the camera
            };

            Debug.Log($"[MOCK RAYCAST] ✓ Hit at {hit.point}");
            return true;
        }
        else
        {
            hit = default;
            Debug.Log($"[MOCK RAYCAST] ✗ Miss");
            return false;
        }
    }
}
