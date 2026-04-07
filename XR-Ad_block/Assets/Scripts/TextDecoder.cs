/*
    TextDecoder.cs

    PURPOSE:
    Converts a PaddleOCR output tensor into a human-readable string.
    
    ARCHITECTURE:
    - Uses a Tensor<float> from TextRecognition.cs
    - Uses a yml file to map class indices to characters
    - Decodes the tensor into a string based on the highest confidence class at each position
    

    IMPORTANT:
    
*/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Unity.InferenceEngine;
using UnityEngine;

public class TextDecoder
{
    private Dictionary<int, char> indexToChar;

    public TextDecoder(TextAsset file)
    {
        this.indexToChar = parseYml(file);
    }

    // Check size for debug purposes, remove later
    public int DictionarySize => indexToChar.Count;

    public string decode(Tensor<float> tensor)
    {
        // The output tensor shape is (1, sequence_length, num_classes)
        int numClasses = tensor.shape[2];
        int sequenceLength = tensor.shape[1];

        var decodedString = new StringBuilder(sequenceLength);
        int previousIndex = -1;

        // For each position in the sequence, find the class with the highest confidence
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

            // Skip consecutive duplicates
            if (maxIndex == previousIndex)
            {
                continue;
            }
            previousIndex = maxIndex;

            // Skip blank token (index 0)
            if (maxIndex == 0)
            {
                continue;
            }

            // Map the class index to a character and append to the decoded string
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

    // Parses the inference.yml character_dict section.
    // Each entry is a YAML list item like "  - A" or "  - '!'"
    // Index 0 in the model output is reserved for "blank", so the first entry maps to index 1.
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

            // Stop when we hit a line that isn't a list item (new YAML section or end of file)
            if (!line.StartsWith("  -"))
                break;

            // Extract the character after "  - "
            string value = line.Substring(4).Trim();

            // Remove YAML quoting: '!' → !
            if (value.Length >= 2 && value[0] == '\'' && value[value.Length - 1] == '\'')
            {
                value = value.Substring(1, value.Length - 2);
                // Handle escaped single quote: '''' becomes '
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
