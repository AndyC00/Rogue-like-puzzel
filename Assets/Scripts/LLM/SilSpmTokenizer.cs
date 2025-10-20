using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using UnityEngine;
using SIL.Machine.Tokenization.SentencePiece;

public sealed class SilSpmTokenizer : ISpmTokenizer, IDisposable
{
    private readonly SentencePieceTokenizer _spm;
    private readonly FileStream _spmStream;
    private readonly Dictionary<string, int> _pieceToId = new();
    private readonly List<string> _idToPiece = new();

    public int BosId { get; private set; } = 1;   // <s>
    public int EosId { get; private set; } = 2;   // </s>
    public int UnkId { get; private set; } = 0;   // <unk>
    public int PadId { get; private set; } = 3;   // <pad>

    public SilSpmTokenizer(
        string modelRelativePath = "LLM/tokenizer.model",
        string vocabJsonRelativePath = "LLM/tokenizer.json",
        string specialMapRelativePath = "LLM/special_tokens_map.json")
    {
        string ResolvePath(string rel)
        {
            string p1 = Path.Combine(Application.streamingAssetsPath, rel);
            if (File.Exists(p1)) return p1;
            string p2 = Path.Combine(Application.dataPath, rel);
            if (File.Exists(p2)) return p2;
            throw new FileNotFoundException($"File not found: {rel}", p1);
        }

        var modelPath = ResolvePath(modelRelativePath);
        var vocabJsonPath = ResolvePath(vocabJsonRelativePath);

        _spm = new SentencePieceTokenizer(modelPath);

        BuildVocabFromTokenizerJson(vocabJsonPath);
        TryLoadSpecialTokenIds(ResolvePath, specialMapRelativePath);
    }

    public int[] Encode(string text, bool addBos = false, bool addEos = false)
    {
        var ids = new List<int>(128);
        if (addBos) ids.Add(BosId);

        foreach (var pieceObj in _spm.Tokenize(text))
        {
            var piece = pieceObj.ToString();
            if (_pieceToId.TryGetValue(piece, out int id)) ids.Add(id);
            else ids.Add(UnkId);
        }
        if (addEos) ids.Add(EosId);
        return ids.ToArray();
    }

    public string Decode(ReadOnlySpan<int> ids)
    {
        var pieces = new List<string>(ids.Length);
        foreach (var id in ids)
        {
            string piece = (id >= 0 && id < _idToPiece.Count) ? _idToPiece[id] : "<unk>";
            pieces.Add(piece);
        }
        var joined = string.Concat(pieces);
        return joined.Replace("▁", " ").Trim();
    }

    public void Dispose() { /* no-op */ }

    private void BuildVocabFromTokenizerJson(string tokenizerJsonPath)
    {
        using var fs = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(fs);

        if (doc.RootElement.TryGetProperty("model", out var model) &&
            model.TryGetProperty("vocab", out var vocab))
        {
            int id = 0;
            foreach (var item in vocab.EnumerateArray())
            {
                string piece = item[0].GetString(); // ["piece", score]
                _pieceToId[piece] = id;
                _idToPiece.Add(piece);
                id++;
            }
        }
        if (doc.RootElement.TryGetProperty("added_tokens", out var added))
        {
            foreach (var tok in added.EnumerateArray())
            {
                if (!tok.TryGetProperty("id", out var idProp)) continue;
                if (!tok.TryGetProperty("content", out var contentProp)) continue;
                int id = idProp.GetInt32();
                string piece = contentProp.GetString();
                while (_idToPiece.Count <= id) _idToPiece.Add(string.Empty);
                _idToPiece[id] = piece;
                _pieceToId[piece] = id;
            }
        }
    }

    private void TryLoadSpecialTokenIds(Func<string, string> resolve, string specialMapRelativePath)
    {
        try
        {
            string path = resolve(specialMapRelativePath);
            if (!File.Exists(path)) return;

            using var fs = File.OpenRead(path);
            using var doc = JsonDocument.Parse(fs);

            string Get(string name)
                => doc.RootElement.TryGetProperty(name, out var e) ? e.GetString() : null;

            string bos = Get("bos_token");
            string eos = Get("eos_token");
            string unk = Get("unk_token");
            string pad = Get("pad_token");

            if (bos != null && _pieceToId.TryGetValue(bos, out int bosId)) BosId = bosId;
            if (eos != null && _pieceToId.TryGetValue(eos, out int eosId)) EosId = eosId;
            if (unk != null && _pieceToId.TryGetValue(unk, out int unkId)) UnkId = unkId;
            if (pad != null && _pieceToId.TryGetValue(pad, out int padId)) PadId = padId;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[SilSpmTokenizer] failed to read special_tokens_map: {ex.Message}");
        }
    }
}
