/*
    ScrollViewThumbstick

    PURPOSE:
    Enables scrolling through the Options image ScrollView using the
    right thumbstick on the Meta Quest controller without requiring
    the user to first focus the scroll view with the ray.

    ARCHITECTURE:
    - Reads OVRInput right thumbstick Y axis each Update frame.
    - Applies scroll directly to the ScrollRect's vertical position.
    - Dead zone prevents drift from slight thumbstick movement.

    SETUP:
    - Attach to the ImageScroller GameObject.
    - Assign the ScrollRect reference in the Inspector.
*/

using UnityEngine;
using UnityEngine.UI;

public class ScrollViewThumbstick : MonoBehaviour
{
    [SerializeField] private ScrollRect scrollRect;
    [SerializeField] private float scrollSpeed = 0.5f;
    [SerializeField] private float deadZone = 0.1f;
    private void Update()
    {
        if (scrollRect == null) return;

        // Only scroll when Options panel is active
        if (!gameObject.activeInHierarchy) return;

        float input = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick).y;

        if (Mathf.Abs(input) > deadZone)
        {
            scrollRect.verticalNormalizedPosition = Mathf.Clamp01(scrollRect.verticalNormalizedPosition + input * scrollSpeed * Time.deltaTime);
        }
    }
}
