/*
    Summary:
    Scrolls the options image list with the right thumbstick while the
    options panel is active.

    Pipeline:
    OVRInput -> ScrollViewThumbstick -> ScrollRect
*/

using UnityEngine;
using UnityEngine.UI;

public class ScrollViewThumbstick : MonoBehaviour
{
    [SerializeField]
    private ScrollRect scrollRect;

    [SerializeField]
    private float scrollSpeed = 0.5f;

    [SerializeField]
    private float deadZone = 0.1f;

    private void Update()
    {
        if (scrollRect == null)
            return;

        if (!gameObject.activeInHierarchy)
            return;

        float input = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick).y;

        if (Mathf.Abs(input) > deadZone)
        {
            scrollRect.verticalNormalizedPosition = Mathf.Clamp01(
                scrollRect.verticalNormalizedPosition + input * scrollSpeed * Time.deltaTime
            );
        }
    }
}
