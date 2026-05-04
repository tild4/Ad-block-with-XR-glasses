// <summary>
// evaluates if an object should be blocked or not based on confindence scores
// from OCR

using System;
using System.Collections.Generic;
using System.Linq;
//using System.Diagnostics;
using Meta.XR;
using UnityEngine;

public class DecisionManager : MonoBehaviour
{
    [SerializeField]
    private TextRecognitionInference_MVP2 textRecognitionInference;

    [SerializeField]
    private NLPClassifier_MVP2 nlpClassifier;

    [SerializeField, Range(0f, 1f)]
    private float nlpThreshold = 0.5f;


    // Event invoked when a decision is made for a tracked object: (obj, concatenatedText, shouldBlock)
    public event Action<TrackedObject, string, bool> onDecisionMade;

    private void OnEnable()
    {
        if (textRecognitionInference != null)
        {
            textRecognitionInference.sendTexts += HandleTexts;
        }
    }

    private void OnDisable()
    {
        if (textRecognitionInference != null)
        {
            textRecognitionInference.sendTexts -= HandleTexts;
        }
    }

    // Handle texts produced by TextRecognitionInference pipeline
    public void HandleTexts(TextsPerAd textsPerAd)
    {
        Debug.Log($"[Decision] HandleTexts called for object {textsPerAd.trackedObject?.id}");
        if (textsPerAd.trackedObject == null)
            return;

        string combined = 
        textsPerAd.texts != null ? string.Join(" ", textsPerAd.texts) : string.Empty;
        
        
        nlpClassifier.Classify(combined, (label, probs) =>
        {
            // ocrConfidence = P(non-ad) + P(socially beneficial)
            float ocrConfidence = probs[0] + probs[1];

            var decision = DecisionResult(ocrConfidence, combined);
            onDecisionMade?.Invoke(textsPerAd.trackedObject, combined, decision.shouldBlock);

        });

    }

    // Block when P(non-ad) + P(socially beneficial) is low, i.e. P(ad) is high.
    private (bool shouldBlock, float confidence) DecisionResult(
        float ocrConfidence, string text
    )
    {
        bool shouldBlock = ocrConfidence < nlpThreshold;
        Debug.Log($"[Decision] Evaluating: '{text}' with P(non-ad+beneficial)={ocrConfidence:F2} → shouldBlock={shouldBlock}");
        return (shouldBlock, ocrConfidence);
    }
}
