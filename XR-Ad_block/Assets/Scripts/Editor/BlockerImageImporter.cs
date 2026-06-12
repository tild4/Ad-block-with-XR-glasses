/*
    Summary:
    Configures imported Resources/BlockerImages textures as sprites for the
    runtime Options image picker.

    Pipeline:
    Unity asset import -> BlockerImageImporter -> OptionsUI sprite loading
*/

using UnityEditor;
using UnityEngine;

public class BlockerImageImporter : AssetPostprocessor
{
    private const string TargetFolder = "Resources/BlockerImages";

    void OnPreprocessTexture()
    {
        if (!assetPath.Contains(TargetFolder))
            return;

        TextureImporter importer = (TextureImporter)assetImporter;

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
