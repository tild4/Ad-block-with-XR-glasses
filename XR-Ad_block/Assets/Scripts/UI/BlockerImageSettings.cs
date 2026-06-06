/*
    Summary:
    Stores the currently selected blocker image for new block visuals.

    Pipeline:
    OptionsUI -> BlockerImageSettings -> BlockVisualization
*/

using UnityEngine;

public static class BlockerImageSettings
{
    public static Sprite SelectedSprite { get; set; } = null;
}
