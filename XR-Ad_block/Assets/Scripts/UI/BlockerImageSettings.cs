/*
    BlockerImageSettings

    PURPOSE:
    Lightweight static container that holds the user's currently selected
    blocker image across the lifetime of the application session.

    ARCHITECTURE:
    - Written by OptionsUI when the user picks an image.
    - Read by BlockVisualization when a new block is spawned.
    - SelectedSprite being null means use the default material appearance.
    - Static class — no MonoBehaviour needed, lives for the entire session.
*/

using UnityEngine;

public static class BlockerImageSettings
{
    public static Sprite SelectedSprite { get; set; } = null;
}
