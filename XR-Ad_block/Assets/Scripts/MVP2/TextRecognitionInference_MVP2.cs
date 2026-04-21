/*
    TextRecognitionInference

    PURPOSE:
    Runs OCR recognition on cropped word ROI tensors, decodes the text,
    and logs the recognized output.

    CURRENT FLOW:
    ProcessOCRDetection2 -> THIS

    POLICY:
    - Latest batch wins.
    - Older pending batches are disposed before they start.
    - Recognition runs sequentially per ROI.
*/

using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Rendering;

public class TextRecognitionInference_MVP2 : MonoBehaviour
{
    [SerializeField]
    private ViewCroppedImage viewCroppedImage;

    /*
    OCR recognition input:
    [DynamicDimension.0, 3, 48, DynamicDimension.1] in NCHW format.

    Width can vary, though multiples of 32 are recommended.
    Batch remains 1 per ROI.
    */
    [SerializeField]
    private ProcessOCRDetection_MVP2 getROIText;

    [SerializeField]
    private int tensorTargetHeight = 48;

    [SerializeField]
    private int tensorTargetWidth = 320;

    [SerializeField]
    private Material cropMaterial;

    [SerializeField]
    private ModelAsset modelAsset;

    [SerializeField]
    private TextAsset ymlFile;

    private Worker worker;
    private TextDecoder textDecoder;
    private bool isProcessing = false;

    public event Action<TextsPerAd> sendTexts;

    private void Awake()
    {
        if (modelAsset == null || ymlFile == null)
        {
            Debug.Log("Missing asset");
            return;
        }

        var ocrModel = ModelLoader.Load(modelAsset);

        textDecoder = new TextDecoder(ymlFile);
        worker = new Worker(ocrModel, BackendType.CPU);
    }

    private void OnEnable()
    {
        if (getROIText != null)
        {
            getROIText.sendCroppedROIText += HandleNewTextTensorsPerAd;
        }
    }

    private void OnDisable()
    {
        if (getROIText != null)
        {
            getROIText.sendCroppedROIText -= HandleNewTextTensorsPerAd;
        }
    }

    /*
        Receives a new recognition batch.

        FLOW:
        1. Filter invalid ROI tensors.
        2. Keep only the newest pending batch.
        3. Start processing if needed.
    */
    private void HandleNewTextTensorsPerAd(TextTensorsPerAd advertisement)
    {
        if (advertisement.trackedObject == null || advertisement.textRegions == null)
        {
            return;
        }

        if (!isProcessing)
        {
            StartCoroutine(ProcessAd(advertisement));
        }
    }

    private IEnumerator ProcessAd(TextTensorsPerAd advertisement)
    {
        // Prevents nested coroutines
        isProcessing = true;

        List<string> texts = new List<string>();

        // For every list of text regions per detected ad
        foreach (var textROI in advertisement.textRegions)
        {
            if (textROI.textRegion != null)
            {
                yield return RunInference(textROI, texts);
            }
        }

        TextsPerAd adWithTexts = new TextsPerAd(advertisement.trackedObject, texts);
        sendTexts?.Invoke(adWithTexts);

        isProcessing = false;
    }

    /*
        Runs OCR recognition for one cropped word ROI.

        FLOW:
        1. Schedule inference.
        2. Wait for async readback.
        3. Dispose the input tensor.
        4. Decode and log the recognized text.
    */
    private IEnumerator RunInference(TextTensor textROI, List<string> texts)
    {
        Tensor<float> inputTensor = textROI.textRegion;
        //Rect relativeBounds = textROI.relativeBounds;

        if (inputTensor == null || worker == null)
        {
            yield return null;
            yield break;
        }

        PipelineProfiler.begin("OCR RecPreprocess");
        Tensor<float> normalizedInput = NormalizeForPPOCRRecognition(inputTensor);
        PipelineProfiler.end("OCR RecPreprocess");
        inputTensor.Dispose();

        PipelineProfiler.begin("OCR TextRecog");
        worker.Schedule(normalizedInput);

        var outputAwaiter = (worker.PeekOutput(0) as Tensor<float>)
            .ReadbackAndCloneAsync()
            .GetAwaiter();

        while (!outputAwaiter.IsCompleted)
        {
            yield return null;
        }

        PipelineProfiler.end("OCR TextRecog");
        normalizedInput.Dispose();

        Tensor<float> outputTensor = outputAwaiter.GetResult();

        if (outputTensor == null)
        {
            yield break;
        }

        PipelineProfiler.begin("Text decoder");
        string output = textDecoder.decode(outputTensor);
        PipelineProfiler.end("Text decoder");

        outputTensor.Dispose();
        Debug.Log($"[RECOGNITION] Object {textROI} recognized word: '{output}'");

        texts.Add(output);

        viewCroppedImage?.SetDetectedWord(output);
    }

    /*
        Normalizes an RGB [0,1] tensor for PP-OCR text recognition.
        Recognition expects BGR channel order and values scaled to [-1, 1].
    */
    private Tensor<float> NormalizeForPPOCRRecognition(Tensor<float> rgbTensor)
    {
        Tensor<float> cpuTensor = rgbTensor.ReadbackAndClone();
        int height = cpuTensor.shape[2];
        int width = cpuTensor.shape[3];

        Tensor<float> result = new Tensor<float>(new TensorShape(1, 3, height, width));

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float r = cpuTensor[0, 0, y, x];
                float g = cpuTensor[0, 1, y, x];
                float b = cpuTensor[0, 2, y, x];

                result[0, 0, y, x] = (b - 0.5f) / 0.5f;
                result[0, 1, y, x] = (g - 0.5f) / 0.5f;
                result[0, 2, y, x] = (r - 0.5f) / 0.5f;
            }
        }

        cpuTensor.Dispose();
        return result;
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
