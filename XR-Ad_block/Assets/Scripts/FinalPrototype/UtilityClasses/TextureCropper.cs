/*
    Summary:
    Crops normalized texture regions into render textures using the shared
    crop material used by YOLO and OCR preprocessing.
*/

using UnityEngine;

public static class TextureCropper
{
    public static Rect TopLeftToUvRect(Rect topLeftRect)
    {
        float uvY = 1f - topLeftRect.yMax;
        return new Rect(topLeftRect.x, uvY, topLeftRect.width, topLeftRect.height);
    }

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
