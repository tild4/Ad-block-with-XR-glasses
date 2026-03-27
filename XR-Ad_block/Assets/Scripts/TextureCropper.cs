using UnityEngine;

public static class TextureCropper 
{
    //crops the region corresponding to a bounding box using pixel coordinates
    public static Texture2D CropBoundingBox(Rect boundingBox, Texture source)
    {

        if (source == null)
        {
            Debug.LogWarning("TextureCropper: source texture is null.");
            return null;
        }

        int x = (int)boundingBox.x;
        int y = (int)boundingBox.y;
        int width = (int)boundingBox.width;
        int height = (int)boundingBox.height;

        if (width <= 0 || height <= 0)
        {
            Debug.LogWarning($"TextureCropper: invalid crop size ({width}x{height}).");
            return null;
        }

        //create destination texture with same size as ROI
        Texture2D dst = new Texture2D(width, height);
        Graphics.CopyTexture(
            source,      
            0,                  // source mipmap level, not used
            0,                  // source element, not used
            x, y, width, height,// source ROI
            dst, // output texture
            0,                  // dst mipmap level, not used
            0,                  // dst element, not used
            0, 0                // crop offset in dst, 0 because dst and ROI are same size
        );

        return dst;
    }

}
