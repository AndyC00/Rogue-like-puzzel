using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;

using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

[DisallowMultipleComponent]
public class TinyLlamaChatbot : MonoBehaviour
{
    [Header("UI")]
    [SerializeField] private TMP_InputField input;
    [SerializeField] private Button sendBtn;
    [SerializeField] private TextMeshProUGUI chatView;

    [Header("Runtime Model (.sentis in StreamingAssets)")]
    [SerializeField] private string sentisFileName = "model_fp16.sentis";

    [Header("Gen Settings")]
    [Range(0.2f, 2.0f)][SerializeField] private float temperature = 0.9f;
    [Range(0, 100)][SerializeField] private int topK = 40;
    [Range(0.0f, 1.0f)][SerializeField] private float topP = 0.9f;
    [SerializeField] private int maxNewTokens = 64;
    [SerializeField] private int maxContextTokens = 512;

    [Header("ONNX/Sentis I/O Names")]
    [SerializeField] private string inputIdsName = "input_ids";
    [SerializeField] private string attnMaskName = "attention_mask";
    [SerializeField] private string logitsName = "logits";

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

        // 2) load .sentis from StreamingAssets 
        var path = Path.Combine(Application.streamingAssetsPath, sentisFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException($".sentis model not found: {path}");
        _model = ModelLoader.Load(path);
        _worker = new Worker(_model, BackendType.CPU);

        if (!System.IO.File.Exists(path))   // Debug use
        {
            foreach (var f in System.IO.Directory.GetFiles(Application.streamingAssetsPath, "*.sentis"))
                Debug.Log("[LLM] found: " + f);
            throw new System.IO.FileNotFoundException(".sentis model not found: " + path);
        }

        // 3) bond UI
        sendBtn.onClick.AddListener(OnSend);
        LogLine("Chatbot ready. Type and click Send.");
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
        LogLine($"{userText}");
        _ = GenerateAndShowAsync(userText);
    }

    async Task GenerateAndShowAsync(string userText)
    {
        var promptTokens = BuildPromptTokens(userText);
        var newTokens = await Task.Run(() => GenerateTokens(promptTokens, maxNewTokens));
        var reply = _tok.Decode(newTokens.ToArray());
        LogLine($"{reply}");
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

            var idsArr64 = workSeq.Select(i => (long)i).ToArray();
            var maskArr64 = Enumerable.Repeat(1L, seqLen).ToArray();

            using var inputIds = new Tensor<long>(new TensorShape(1, seqLen), idsArr64);
            using var attnMask = new Tensor<long>(new TensorShape(1, seqLen), maskArr64);

            _worker.SetInput(inputIdsName, inputIds);
            if (!string.IsNullOrEmpty(attnMaskName))
                _worker.SetInput(attnMaskName, attnMask);

            _worker.Schedule();

            var logitsTensor = _worker.PeekOutput(logitsName) as Tensor<float>;
            if (logitsTensor == null) { Debug.LogError($"[LLM] Output '{logitsName}' not found or wrong type"); return newTokens; }

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