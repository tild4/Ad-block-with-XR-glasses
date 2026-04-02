using UnityEngine;

public static class TextureCropper 
{

    private static Material cropMat;
    private static Shader cropShader;
    //crops the region corresponding to a bounding box 
    public static bool CropBoundingBox(Rect boundingBox, Texture source, RenderTexture target)
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

        if (cropShader == null)
        {
            cropShader = Shader.Find("Custom/CropShader");
            if (cropShader == null)
            {
                Debug.LogError("TextureCropper: Could not find shader Custom/CropShader.");
                return false;
            }
        }

        if (cropMat == null)
        {
            cropMat = new Material(cropShader);
        }

        cropMat.SetVector("_Crop", new Vector4(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height));
        Graphics.Blit(source, target, cropMat);

        return true;
    }

    public static void Cleanup()
    {
        if (cropMat != null)
        {
            Object.Destroy(cropMat);
            cropMat = null;
        }

        cropShader = null;
    }

}
