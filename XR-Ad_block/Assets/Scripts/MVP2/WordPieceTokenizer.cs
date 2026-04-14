/*

    PURPOSE:
    Converts a raw string (from OCR) into token IDs for the NLP model (ELECTRA ONNX) 
    becasue the model expects Tensor<int> input, not raw text. 


    CURRENT FLOW:
    

    POLICY:
    - Latest batch wins.
    - The class owns incoming tensors and disposes dropped or consumed ones.
    - Processing yields during heavy CPU work to avoid monopolizing the frame.

    NOTE:
    

*/

using System.Collections.Generic;
using UnityEngine;

public readonly struct TokenizedResult
{
    public readonly int[] inputIds;
    public readonly int[] attentionMask;

    public TokenizedResult(int[] inputIds, int[] attentionMask)
    {
        this.inputIds = inputIds;
        this.attentionMask = attentionMask;
    }
}

public class WordPieceTokenizer
{
    private readonly Dictionary<string, int> vocab;

    private readonly int padId;
    private readonly int unkId;
    private readonly int clsId;
    private readonly int sepId;

    private readonly int maxLength;

    public WordPieceTokenizer(TextAsset vocabAsset, int maxLength = 64)
    {
        this.maxLength = maxLength;
        vocab = new Dictionary<string, int>();

        string[] lines = vocabAsset.text.Split('\n');

        for (int i = 0; i < lines.Length; i++)
        {
            string token = lines[i].TrimEnd('\r');
            if (token.Length > 0)
            {
                vocab[token] = i;
            }
        }

        padId = LookupOrThrow("[PAD]");
        unkId = LookupOrThrow("[UNK]");
        clsId = LookupOrThrow("[CLS]");
        sepId = LookupOrThrow("[SEP]");

        Debug.Log($"[WordPiece] Loaded {vocab.Count} tokens. PAD={padId}, UNK={unkId}, CLS={clsId}, SEP={sepId}");
    }

    public TokenizedResult Tokenize(string text)
    {
        // 1. Splitta till ord
        List<string> words = PreTokenize(text);

        // 2. WordPiece varje ord → samla alla token-IDs
        var allIds = new List<int>();
        foreach (string word in words)
        {
            allIds.AddRange(WordPieceTokenizeWord(word));
        }

        // 3. Trunkera om för långt (behåll plats för [CLS] och [SEP])
        int maxTokens = maxLength - 2;
        if (allIds.Count > maxTokens)
        {
            allIds.RemoveRange(maxTokens, allIds.Count - maxTokens);
        }

        // 4. Bygg sekvensen: [CLS] tokens... [SEP] [PAD] [PAD]...
        int[] inputIds = new int[maxLength];
        int[] attentionMask = new int[maxLength];

        inputIds[0] = clsId;
        attentionMask[0] = 1;

        for (int i = 0; i < allIds.Count; i++)
        {
            inputIds[i + 1] = allIds[i];
            attentionMask[i + 1] = 1;
        }

        int sepPos = allIds.Count + 1;
        inputIds[sepPos] = sepId;
        attentionMask[sepPos] = 1;

        // Resten är redan 0 (padId=0, attentionMask=0) tack vare int[] default
        return new TokenizedResult(inputIds, attentionMask);
    }

    private int LookupOrThrow(string token)
    {
        if (vocab.TryGetValue(token, out int id))
        {
            return id;
        }
        throw new System.Exception($"[WordPiece] Special token '{token}' not found in vocab.txt");
    }

    private List<string> PreTokenize(string text)
    {
        var words = new List<string>();
        int i = 0;

        while (i < text.Length)
        {
            char c = text[i];

            // Skip whitespace
            if (char.IsWhiteSpace(c))
            {
                i++;
                continue;
            }

            // Punctuation becomes its own token
            if (char.IsPunctuation(c) || char.IsSymbol(c))
            {
                words.Add(c.ToString());
                i++;
                continue;
            }

            // Collect consecutive letters/digits as one word
            int start = i;
            while (i < text.Length && !char.IsWhiteSpace(text[i])
                    && !char.IsPunctuation(text[i]) && !char.IsSymbol(text[i]))
            {
                i++;
            }
            words.Add(text.Substring(start, i - start));
        }

        return words;
    }

    private List<int> WordPieceTokenizeWord(string word)
    {
        var tokenIds = new List<int>();
        int start = 0;

        while (start < word.Length)
        {
            int end = word.Length;
            bool found = false;

            // Greedy longest-match
            while (start < end)
            {
                string substr = word.Substring(start, end - start);

                // All partial matches get the "##" prefix, except the very first one
                if (start > 0)
                {
                    substr = "##" + substr;
                }

                if (vocab.ContainsKey(substr))
                {
                    tokenIds.Add(vocab[substr]);
                    found = true;
                    break;
                }

                end--;
            }

            // If no match found, use [UNK] 
            if (!found)
            {
                tokenIds.Add(unkId);
                break;
            }

            start = end;
        }

        return tokenIds;
    }

}





