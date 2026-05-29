/*
    BlockerImageImporter

    PURPOSE:
    Automatically configures any texture imported into the
    Resources/BlockerImages folder with the correct settings
    for use as a UI sprite in the Options menu.

    ARCHITECTURE:
    - Extends AssetPostprocessor which Unity calls automatically
      on every asset import.
    - Checks if the imported asset is inside Resources/BlockerImages.
    - If so, sets Texture Type to Sprite (2D and UI) and
      Sprite Mode to Single before the import completes.
    - Runs in the Editor only — no runtime overhead.

    SETUP:
    - Place this script in any folder named Editor inside Assets.
    - No further configuration needed. Drop images into
      Assets/Resources/BlockerImages and they will be
      automatically configured on import.
*/

using UnityEngine;
using UnityEditor;

public class BlockerImageImporter : AssetPostprocessor
{
    private const string TargetFolder = "Resources/BlockerImages";

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
