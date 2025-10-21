using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class TinyLlamaChatbot : MonoBehaviour
{
    [Header("UI")]
    public InputField input;
    public Button sendBtn;
    public Text chatView;

    [Header("Model (assign .onnx asset here)")]
    public ModelAsset modelAsset;

    [Header("Gen Settings")]
    [Range(0.2f, 2.0f)] public float temperature = 0.9f;
    [Range(0, 100)] public int topK = 40;
    [Range(0.0f, 1.0f)] public float topP = 0.9f;
    public int maxNewTokens = 64;
    public int maxContextTokens = 512;

    [Header("ONNX I/O Names")]
    public string inputIdsName = "input_ids";
    public string attnMaskName = "attention_mask";
    public string logitsName = "logits";

    private int _bosId, _eosId, _padId;
    private Model _model;
    private Worker _worker;
    private SilSpmTokenizer _tok;
    private readonly List<int> _context = new();

    void Awake()
    {
        // 1) tokenizer
        _tok = new SilSpmTokenizer();
        _bosId = _tok.BosId; _eosId = _tok.EosId; _padId = _tok.PadId;

        // 2) load model via ModelAsset
        if (modelAsset == null) throw new Exception("Assign ModelAsset (.onnx) in Inspector.");
        _model = ModelLoader.Load(modelAsset);
        _worker = new Worker(_model, BackendType.CPU);

        // 3) UI
        sendBtn.onClick.AddListener(OnSend);
        LogLine("✅ Chatbot ready. Type and click Send.");
    }

    void OnDestroy()
    {
        _worker?.Dispose();
        _tok?.Dispose();
    }

    void OnSend()
    {
        var userText = input.text?.Trim();
        if (string.IsNullOrEmpty(userText)) return;
        input.text = "";
        LogLine($"👤 {userText}");
        _ = GenerateAndShowAsync(userText);
    }

    async Task GenerateAndShowAsync(string userText)
    {
        var promptTokens = BuildPromptTokens(userText);
        var newTokens = await Task.Run(() => GenerateTokens(promptTokens, maxNewTokens));
        var reply = _tok.Decode(newTokens.ToArray());
        LogLine($"🤖 {reply}");
    }

    List<int> BuildPromptTokens(string userText)
    {
        var tokens = _tok.Encode(userText, addBos: false, addEos: false).ToList();
        var merged = new List<int>(_context.Count + tokens.Count + 4);
        if (_bosId >= 0) merged.Add(_bosId);
        merged.AddRange(_context);
        merged.AddRange(tokens);
        if (merged.Count > maxContextTokens)
            merged = merged.Skip(merged.Count - maxContextTokens).ToList();
        return merged;
    }

    List<int> GenerateTokens(List<int> inputTokens, int maxToGen)
    {
        var workSeq = new List<int>(inputTokens);
        var newTokens = new List<int>();

        for (int step = 0; step < maxToGen; step++)
        {
            int seqLen = workSeq.Count;

            var idsArr = workSeq.ToArray();
            var maskArr = Enumerable.Repeat(1, seqLen).ToArray();

            using var inputIds = new Tensor<int>(new TensorShape(new[] { 1, seqLen }), idsArr);
            using var attnMask = new Tensor<int>(new TensorShape(new[] { 1, seqLen }), maskArr);

            _worker.SetInput(inputIdsName, inputIds);
            _worker.SetInput(attnMaskName, attnMask);
            _worker.Schedule();

            var logitsTensor = _worker.PeekOutput(logitsName) as Tensor<float>;
            logitsTensor.CompleteAllPendingOperations();
            var raw = logitsTensor.DownloadToArray();
            var shape = logitsTensor.shape; // [1, seq, vocab]
            int vocab = shape[2];

            int start = (seqLen - 1) * vocab;
            var lastLogits = new float[vocab];
            Array.Copy(raw, start, lastLogits, 0, vocab);

            int nextId = SampleFromLogits(lastLogits, temperature, topK, topP);
            if (_eosId >= 0 && nextId == _eosId) break;
            newTokens.Add(nextId);
            workSeq.Add(nextId);
        }

        _context.AddRange(_tok.Encode(" "));
        _context.AddRange(newTokens);
        if (_context.Count > maxContextTokens)
            _context.RemoveRange(0, _context.Count - maxContextTokens);

        return newTokens;
    }

    int SampleFromLogits(float[] logits, float temp, int topK, float topP)
    {
        var scaled = new float[logits.Length];
        float maxLogit = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i] / Mathf.Max(1e-6f, temp);
            scaled[i] = v;
            if (v > maxLogit) maxLogit = v;
        }
        double sum = 0;
        for (int i = 0; i < scaled.Length; i++) { scaled[i] = (float)Math.Exp(scaled[i] - maxLogit); sum += scaled[i]; }
        for (int i = 0; i < scaled.Length; i++) scaled[i] = (float)(scaled[i] / sum);

        var idx = Enumerable.Range(0, scaled.Length).ToArray();
        Array.Sort(scaled, idx); Array.Reverse(scaled); Array.Reverse(idx);

        int keep = (topK > 0) ? Math.Min(topK, scaled.Length) : scaled.Length;
        float c = 0; int cut = keep;
        if (topP > 0 && topP < 1.0f)
        {
            for (int i = 0; i < keep; i++) { c += scaled[i]; if (c >= topP) { cut = i + 1; break; } }
        }

        float total = 0f; for (int i = 0; i < cut; i++) total += scaled[i];
        float r = UnityEngine.Random.Range(0f, total), acc = 0f;
        for (int i = 0; i < cut; i++) { acc += scaled[i]; if (r <= acc) return idx[i]; }
        return idx[0];
    }

    void LogLine(string s)
    {
        if (chatView) chatView.text += s + "\n";
        else Debug.Log(s);
    }
}
