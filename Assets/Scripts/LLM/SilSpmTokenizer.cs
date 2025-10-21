using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using UnityEngine;
using Microsoft.ML.Tokenizers;

public sealed class SilSpmTokenizer : ISpmTokenizer, IDisposable
{
    private SentencePieceTokenizer _sp;
    private readonly Stream _modelStream;

    public int BosId { get; private set; } = 1; // <s>
    public int EosId { get; private set; } = 2; // </s>
    public int UnkId { get; private set; } = 0; // <unk>
    public int PadId { get; private set; } = 3; // <pad>

    public SilSpmTokenizer(
        string modelRelativePath = "LLM/tokenizer.model",
        string specialMapRelativePath = "LLM/special_tokens_map.json")
    {
        string pA = Path.Combine(Application.streamingAssetsPath, modelRelativePath);
        string pB = Path.Combine(Application.dataPath, modelRelativePath);
        string spmPath = File.Exists(pA) ? pA : (File.Exists(pB) ? pB : null);
        if (spmPath == null) throw new FileNotFoundException("SentencePiece model not found", modelRelativePath);

        TryLoadSpecialTokenIds(specialMapRelativePath);

        _modelStream = File.OpenRead(spmPath);

        _sp = LlamaTokenizer.Create(
            _modelStream,
            addBeginOfSentence: false,
            addEndOfSentence: false
        );

    }

    public int[] Encode(string text, bool addBos = false, bool addEos = false)
    {
        var ids = _sp.EncodeToIds(text, addBos, addEos);
        return ids is int[] a ? a : ToArrayFast(ids);
    }

    public string Decode(ReadOnlySpan<int> ids)
    {
        var src = ids.ToArray();
        var filtered = new List<int>(src.Length);
        for (int i = 0; i < src.Length; i++)
        {
            int id = src[i];
            if (id == BosId || id == EosId || id == PadId) continue;
            filtered.Add(id);
        }
        return _sp.Decode(filtered);
    }

    private static int[] ToArrayFast(IReadOnlyList<int> list)
    {
        var arr = new int[list.Count];
        for (int i = 0; i < list.Count; i++) arr[i] = list[i];
        return arr;
    }

    private void TryLoadSpecialTokenIds(string specialMapRelativePath)
    {
        string pA = Path.Combine(Application.streamingAssetsPath, specialMapRelativePath);
        string pB = Path.Combine(Application.dataPath, specialMapRelativePath);
        string path = File.Exists(pA) ? pA : (File.Exists(pB) ? pB : null);
        if (path == null) return;

        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        if (root.TryGetProperty("bos_token_id", out var b) && b.ValueKind == JsonValueKind.Number) BosId = b.GetInt32();
        if (root.TryGetProperty("eos_token_id", out var e) && e.ValueKind == JsonValueKind.Number) EosId = e.GetInt32();
        if (root.TryGetProperty("unk_token_id", out var u) && u.ValueKind == JsonValueKind.Number) UnkId = u.GetInt32();
        if (root.TryGetProperty("pad_token_id", out var p) && p.ValueKind == JsonValueKind.Number) PadId = p.GetInt32();
    }

    public void Dispose() => _modelStream?.Dispose();
}
