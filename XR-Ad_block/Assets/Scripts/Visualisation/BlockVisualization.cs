/*
    BlockVisualization

    PURPOSE:
    Manages the individual visual representation of a single ad-blocker.
    
    ARCHITECTURE:
    - Initialization: Scales up the block with a "Pop-in" animation.
    - UI Update: Sets the TextMeshPro text to show the object's unique ID.
    - Feedback: Includes a warning state (red blink) to notify when
      the object is about to be deleted.

    IMPORTANT:
    Uses URP-specific material properties ("_BaseColor") to change
    visuals at runtime without breaking performance.
*/
using TMPro;
using UnityEngine;

public class BlockVisualization : MonoBehaviour
{
    [Header("UI Elements")]
    [SerializeField]
    private TextMeshPro idText;

    /*
        Called to initialize the block's visual data, specifically setting the ID text.
    */
    public void SetBlockData(int id)
    {
        if (idText != null)
        {
            idText.text = $"ID: {id}";
        }
    }
}
