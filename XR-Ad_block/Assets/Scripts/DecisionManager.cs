// <summary>
// evaluates if an object should be blocked or not based on confindence scores
// from yolo model and OCR

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

    [SerializeField]
    private bool useNLPClassifier = true; // Set to true to enable NLP classification in the decision process]

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

        float yoloConfidence = textsPerAd.trackedObject.lastDetection.confidence;

        // NLP path: replace heuristic with classifier
        if (useNLPClassifier && nlpClassifier != null && !string.IsNullOrWhiteSpace(combined))
        {
            nlpClassifier.Classify(combined, (label, probs) =>
            {
                // ocrConfidence = P(reklam) + P(skadlig) + P(samhällsnyttig)
                float ocrConfidence = probs[1] + probs[2] + probs[3];

                var decision = DecisionResult(yoloConfidence, ocrConfidence);
                onDecisionMade?.Invoke(textsPerAd.trackedObject, combined, decision.shouldBlock);

                Debug.Log(
                    $"[Decision] Object {textsPerAd.trackedObject.id}: NLP='{label}', "
                    + $"ocrConf={ocrConfidence:F2}, shouldBlock={decision.shouldBlock}"
                );
            });
        } 
        else
        {
            // If not using NLP, fall back to heuristic decision based on YOLO confidence and OCR confidence
            float ocrConfidence = 0f;
            if (textsPerAd.texts != null && textsPerAd.texts.Count > 0)
            {
                int nonEmpty = textsPerAd.texts.Count(t => !string.IsNullOrWhiteSpace(t));
                ocrConfidence = nonEmpty > 0 ? 0.9f : 0f;
            }

            var decision = DecisionResult(yoloConfidence, ocrConfidence);
            onDecisionMade?.Invoke(textsPerAd.trackedObject, combined, decision.shouldBlock);

            Debug.Log(
                $"[Decision] Object {textsPerAd.trackedObject.id}: Heuristic decision, "
                + $"ocrConf={ocrConfidence:F2}, shouldBlock={decision.shouldBlock}"
            );
        }

    }

    private (bool shouldBlock, float confidence) DecisionResult(
        float yoloConfidence,
        float ocrConfidence
    )
    {
        // initializing thresholds for blocking
        float upperThreshold = 0.8f;
        float fusionThreshold = 0.7f;
        float lowerThreshold = 0.4f;

        //initializing wheights for confidence fusion
        float yoloWeight = 0.6f;
        float ocrWeight = 0.4f;

        //calculating fusion confidence
        float confidence = yoloConfidence * yoloWeight + ocrConfidence * ocrWeight;

        bool shouldBlock;

        //if yolo confidence good enough, block
        if (yoloConfidence > upperThreshold)
        {
            Debug.Log($"Confidence high: YOLO = {yoloConfidence}, should block = true");
            return (true, yoloConfidence);
        }
        // if yolo model uncertain, use ocr confidence and confidence fusion
        else if (yoloConfidence > lowerThreshold && yoloConfidence < upperThreshold)
        {
            shouldBlock = confidence > fusionThreshold;
            Debug.Log(
                $"Intermediate confidence: fusion confidence = {confidence}, should block:{shouldBlock}"
            );
            return (shouldBlock, confidence);
        }
        // if yolo confidence low, do not block
        else
        {
            Debug.Log($"Confidence low: YOLO = {yoloConfidence}, should block = false");
            return (false, yoloConfidence);
        }
    }

    //lite osäker på om denna del fungerar rätt
    public void Evaluate(float yoloConfidence, float ocrConfidence, Rect rect, FrameData frame)
    {
        var result = DecisionResult(yoloConfidence, ocrConfidence);
        float confidence = result.confidence;
        // This method computes a decision but no longer performs placement.
        // Consumers should subscribe to `onDecisionMade` for full object-level decisions.
    }
}
