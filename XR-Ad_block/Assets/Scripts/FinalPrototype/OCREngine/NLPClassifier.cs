/*
    Summary:
    Classifies combined OCR text with a fine-tuned ELECTRA ONNX model and
    returns category probabilities to the decision stage.

    Pipeline:
    DecisionManager -> NLPClassifier -> DecisionManager callback
*/

using System;
using System.Collections;
using Unity.InferenceEngine;
using UnityEngine;

public class NLPClassifier : MonoBehaviour
{
    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private TextAsset vocabAsset;

    [SerializeField]
    private int maxLength = 64;

    private Worker worker;
    private WordPieceTokenizer tokenizer;
    private bool isProcessing = false;

    private static readonly string[] Labels = { "non-ad", "socially beneficial", "ad" };

    private void Awake()
    {
        if (modelAsset == null || vocabAsset == null)
        {
            Debug.LogError("[NLPClassifier] Missing modelAsset or vocabAsset");
            return;
        }

        var model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.CPU);
        tokenizer = new WordPieceTokenizer(vocabAsset, maxLength);

        Debug.Log("[NLPClassifier] Ready.");
    }

    public void Classify(string text, Action<string, float[]> onResult)
    {
        if (isProcessing || worker == null)
        {
            return;
        }

        StartCoroutine(RunClassification(text, onResult));
    }

    private IEnumerator RunClassification(string text, Action<string, float[]> onResult)
    {
        isProcessing = true;

        TokenizedResult tokens = tokenizer.Tokenize(text);

        var shape = new TensorShape(1, maxLength);
        var inputIds = new Tensor<int>(shape, tokens.inputIds);
        var attentionMask = new Tensor<int>(shape, tokens.attentionMask);
        var tokenTypeIds = new Tensor<int>(shape, new int[maxLength]);

        PipelineProfiler.begin("NLP Classify");
        worker.SetInput("input_ids", inputIds);
        worker.SetInput("attention_mask", attentionMask);
        worker.SetInput("token_type_ids", tokenTypeIds);
        worker.Schedule();

        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>)
            .ReadbackAndCloneAsync()
            .GetAwaiter();

        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        PipelineProfiler.end("NLP Classify");

        inputIds.Dispose();
        attentionMask.Dispose();
        tokenTypeIds.Dispose();

        Tensor<float> output = outputAwaiter.GetResult();
        if (output == null)
        {
            isProcessing = false;
            yield break;
        }

        float[] probs = new float[Labels.Length];
        float sumExp = 0f;

        for (int i = 0; i < Labels.Length; i++)
        {
            probs[i] = Mathf.Exp(output[0, i]);
            sumExp += probs[i];
        }
        for (int i = 0; i < Labels.Length; i++)
        {
            probs[i] /= sumExp;
        }

        output.Dispose();

        string bestLabel = Labels[0];
        for (int i = 1; i < Labels.Length; i++)
        {
            if (probs[i] > probs[System.Array.IndexOf(Labels, bestLabel)])
                bestLabel = Labels[i];
        }

        Debug.Log(
            $"[NLPClassifier] '{text}' -> {bestLabel} ({probs[System.Array.IndexOf(Labels, bestLabel)]:P0})"
        );
        onResult?.Invoke(bestLabel, probs);

        isProcessing = false;
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
