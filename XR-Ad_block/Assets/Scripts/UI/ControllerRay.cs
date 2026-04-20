using UnityEngine;

public class ControllerRay : MonoBehaviour
{
    [SerializeField] private Color rayColor = new Color(1f, 1f, 1f, 0.5f);
    [SerializeField] private float rayLength = 5f;
    [SerializeField] private Material rayMaterial;

    private LineRenderer _line;

    private void Awake()
    {
        _line = gameObject.AddComponent<LineRenderer>();
        _line.positionCount = 2;
        _line.startWidth = 0.003f;
        _line.endWidth = 0.001f;
        _line.useWorldSpace = true;
        _line.material = rayMaterial;
        _line.startColor = rayColor;
        _line.endColor = new Color(rayColor.r, rayColor.g, rayColor.b, 0f);
    }

    private void Update()
    {
        Vector3 origin = transform.position;
        Vector3 direction = transform.forward;
        float endDistance = rayLength;

        //Check if ray hits any UI via physics raycast
        if (Physics.Raycast(origin, direction, out RaycastHit hit, rayLength))
        {
            endDistance = hit.distance;
        }

        _line.SetPosition(0, origin);
        _line.SetPosition(1, origin + direction * endDistance);
    }
}
