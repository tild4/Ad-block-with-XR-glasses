/*
    Summary:
    Converts OCR text into WordPiece token IDs and attention masks for the
    ELECTRA NLP classifier.

    Pipeline:
    NLPClassifier -> WordPieceTokenizer -> NLPClassifier tensors
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

        Debug.Log(
            $"[WordPiece] Loaded {vocab.Count} tokens. PAD={padId}, UNK={unkId}, CLS={clsId}, SEP={sepId}"
        );
    }

    public TokenizedResult Tokenize(string text)
    {
        List<string> words = PreTokenize(text);

        var allIds = new List<int>();
        foreach (string word in words)
        {
            allIds.AddRange(WordPieceTokenizeWord(word));
        }

        int maxTokens = maxLength - 2;
        if (allIds.Count > maxTokens)
        {
            allIds.RemoveRange(maxTokens, allIds.Count - maxTokens);
        }

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

            if (char.IsPunctuation(c) || char.IsSymbol(c))
            {
                words.Add(c.ToString());
                i++;
                continue;
            }

            int start = i;
            while (
                i < text.Length
                && !char.IsWhiteSpace(text[i])
                && !char.IsPunctuation(text[i])
                && !char.IsSymbol(text[i])
            )
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

            while (start < end)
            {
                string substr = word.Substring(start, end - start);

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
