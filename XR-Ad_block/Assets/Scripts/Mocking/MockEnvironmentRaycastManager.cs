using UnityEngine;
using Meta.XR;

/// <summary>
/// Mock raycast for testing in Editor without Quest.
/// Always returns a hit at specified distance.
/// </summary>
public class MockEnvironmentRaycastManager : MonoBehaviour
{
    [Header("Mock Settings")]
    [SerializeField] private float hitDistance = 2f;
    [SerializeField] private bool alwaysHit = true;
    
    public bool Raycast(Ray ray, out EnvironmentRaycastHit hit)
    {
        if (alwaysHit)
        {
            hit = new EnvironmentRaycastHit
            {
                point = ray.origin + ray.direction * hitDistance,
                normal = -ray.direction
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