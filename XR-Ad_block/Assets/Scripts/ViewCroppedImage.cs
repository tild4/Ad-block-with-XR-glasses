using UnityEngine;
using UnityEngine.UI;

public class ViewCroppedImage : MonoBehaviour
{
    [SerializeField] private RawImage debugImage;

    public void Show(Texture texture)
    {
        if (debugImage == null)
        {
            Debug.LogWarning("CropDebugUI: debugImage is not assigned.");
            return;
        }

        debugImage.texture = texture;
        debugImage.enabled = texture != null;
    }

    public void Clear()
    {
        if (debugImage == null)
        {
            return;
        }

        debugImage.texture = null;
        debugImage.enabled = false;
    }
}
