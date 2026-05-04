using UnityEngine;
using UnityEditor;

public class BlockerImageImporter : AssetPostprocessor
{
    private const string TargetFolder = "Assets/BlockerImages";

    void OnPreprocessTexture()
    {
        // Only process textures in the target folder
        if (!assetPath.Contains(TargetFolder))
            return;

        TextureImporter importer = (TextureImporter)assetImporter;

        // Only auto-configure on first import, not on reimports
        // This allows manual overrides after the initial import if needed

        if (importer.importSettingsMissing)
        {
            importer.textureType = TextureImporterType.Sprite;
            importer.spriteImportMode = SpriteImportMode.Single;
            importer.mipmapEnabled = false;
            importer.alphaIsTransparency = true;

            Debug.Log($"[BlockerImageImporter] Auto-configured import settings for {assetPath}");
        }
    }
}
