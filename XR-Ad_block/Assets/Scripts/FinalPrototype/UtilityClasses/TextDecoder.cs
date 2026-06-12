/*
    Summary:
    Decodes a PP-OCR recognition output tensor into text using the
    character dictionary from the model YAML file.

    Pipeline:
    TextRecognitionInference -> TextDecoder -> recognized string
*/
using System.Collections.Generic;
using System.Text;
using Unity.InferenceEngine;
using UnityEngine;

public class TextDecoder
{
    private Dictionary<int, char> indexToChar;
    private bool hasLoggedClassMismatch;

    public TextDecoder(TextAsset file)
    {
        this.indexToChar = parseYml(file);
    }

    public string decode(Tensor<float> tensor)
    {
        int numClasses = tensor.shape[2];
        int sequenceLength = tensor.shape[1];

        if (!hasLoggedClassMismatch && numClasses != indexToChar.Count + 1)
        {
            hasLoggedClassMismatch = true;
            Debug.LogWarning(
                $"[TextDecoder] Model outputs {numClasses} classes but YAML maps {indexToChar.Count + 1} including blank. Unmapped special classes will be ignored."
            );
        }

        var decodedString = new StringBuilder(sequenceLength);
        int previousIndex = -1;

        for (int i = 0; i < sequenceLength; i++)
        {
            float maxConfidence = float.MinValue;
            int maxIndex = -1;

            for (int j = 0; j < numClasses; j++)
            {
                float confidence = tensor[0, i, j];
                if (confidence > maxConfidence)
                {
                    maxConfidence = confidence;
                    maxIndex = j;
                }
            }

            if (maxIndex == previousIndex)
            {
                continue;
            }
            previousIndex = maxIndex;

            if (maxIndex == 0)
            {
                continue;
            }

            if (maxIndex > indexToChar.Count)
            {
                continue;
            }

            if (indexToChar.TryGetValue(maxIndex, out char character))
            {
                decodedString.Append(character);
            }
            else
            {
                decodedString.Append('?');
            }
        }

        return decodedString.ToString();
    }

    private Dictionary<int, char> parseYml(TextAsset file)
    {
        var map = new Dictionary<int, char>();
        string[] lines = file.text.Split('\n');

        bool inDict = false;
        int charIndex = 1; // index 0 is blank

        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i].TrimEnd();

            if (line.Trim() == "character_dict:")
            {
                inDict = true;
                continue;
            }

            if (!inDict)
                continue;

            if (!line.StartsWith("  -"))
                break;

            string value = line.Substring(4).Trim();

            if (value.Length >= 2 && value[0] == '\'' && value[value.Length - 1] == '\'')
            {
                value = value.Substring(1, value.Length - 2);
                value = value.Replace("''", "'");
            }

            if (value.Length > 0)
            {
                map[charIndex] = value[0];
                charIndex++;
            }
        }

        return map;
    }
}
