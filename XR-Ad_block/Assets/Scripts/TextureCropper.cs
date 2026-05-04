using UnityEngine;

public static class TextureCropper
{
    public static Rect TopLeftToUvRect(Rect topLeftRect)
    {
        float uvY = 1f - topLeftRect.yMax;
        return new Rect(topLeftRect.x, uvY, topLeftRect.width, topLeftRect.height);
    }

    //crops the region corresponding to a bounding box
    public static bool CropBoundingBox(
        Rect boundingBox,
        Texture source,
        RenderTexture target,
        Material cropMaterial
    )
    {
        if (source == null)
        {
            Debug.LogWarning("TextureCropper: source texture is null.");
            return false;
        }

        if (target == null)
        {
            Debug.LogWarning("TextureCropper: target render texture is null.");
            return false;
        }

        if (cropMaterial == null)
        {
            Debug.LogError("TextureCropper: crop material is null.");
            return false;
        }

        cropMaterial.SetVector(
            "_Crop",
            new Vector4(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height)
        );
        Graphics.Blit(source, target, cropMaterial);

        return true;
    }

    // Crops a box expressed in top-left-origin normalized image coordinates.
    public static bool CropBoundingBoxTopLeft(
        Rect boundingBox,
        Texture source,
        RenderTexture target,
        Material cropMaterial
    )
    {
        return CropBoundingBox(TopLeftToUvRect(boundingBox), source, target, cropMaterial);
    }
}
