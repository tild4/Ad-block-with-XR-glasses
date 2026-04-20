/*
    ControllerRay

    PURPOSE:
    Renders a visible ray line from the right controller anchor,
    stopping at the first physics object hit. Displays a small dot
    at the point of contact so the user knows exactly where their
    controller is pointing when interacting with UI elements.

    ARCHITECTURE:
    - LineRenderer draws the beam from controller tip forward.
    - Physics.Raycast stops the line at the first collider hit,
      which includes the Box Collider added to the World Space Canvas.
    - A small sphere primitive marks the hit point.
    - The dot hides automatically when no surface is hit.

    SETUP:
    - Attach to RightControllerAnchor.
    - Assign a simple Unlit material to the Ray Material slot.
    - The Canvas must have a Box Collider sized to match its dimensions.
*/

using UnityEngine;

public class ControllerRay : MonoBehaviour
{
    [SerializeField] private Color rayColor = new Color(1f, 1f, 1f, 0.5f);
    [SerializeField] private float rayLength = 5f;
    [SerializeField] private Material rayMaterial;

    private LineRenderer _line;
    private GameObject _dot;

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

        // Create dot
        _dot = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        _dot.transform.localScale = Vector3.one * 0.008f;
        Destroy(_dot.GetComponent<Collider>());
        _dot.GetComponent<Renderer>().material = rayMaterial;
        _dot.SetActive(false);
    }

    private void Update()
    {
        Vector3 origin = transform.position;
        Vector3 direction = transform.forward;
        float endDistance = rayLength;
        bool hitSomething = false;

        //Check if ray hits any UI via physics raycast
        if (Physics.Raycast(origin, direction, out RaycastHit hit, rayLength))
        {
            endDistance = hit.distance;
            hitSomething = true;
            _dot.transform.position = hit.point;
        }

        _dot.SetActive(hitSomething);
        _line.SetPosition(0, origin);
        _line.SetPosition(1, origin + direction * endDistance);
    }
}
