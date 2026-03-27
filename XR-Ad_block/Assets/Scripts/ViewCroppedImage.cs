using UnityEngine;
using UnityEngine.UI;

public class ViewCroppedImage : MonoBehaviour
{
    [SerializeField] private RawImage debugImage;
    [SerializeField] private AspectRatioFitter aspectRatioFitter;

    public void Show(Texture texture)
    {
        if (debugImage == null)
        {
            Debug.LogWarning("CropDebugUI: debugImage is not assigned.");
            return;
        }

        debugImage.texture = texture;
        debugImage.enabled = texture != null;

        if (texture != null && aspectRatioFitter != null)
        {
            aspectRatioFitter.aspectRatio = (float)texture.width / texture.height;
        }
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
