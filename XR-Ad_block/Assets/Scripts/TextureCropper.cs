using UnityEngine;

public static class TextureCropper 
{
    //crops the region corresponding to a bounding box 
    public static RenderTexture CropBoundingBox(Rect boundingBox, Texture source)
    {

        if (source == null)
        {
            Debug.LogWarning("TextureCropper: source texture is null.");
            return null;
        }

        
        RenderTexture rt = new RenderTexture(320, 48, 0);
        rt.Create();

        Material cropMat = new Material(Shader.Find("Custom/CropShader"));
        cropMat.SetVector("_Crop", new Vector4(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height));
        Graphics.Blit(source, rt, cropMat);

        return rt;
    }

}
