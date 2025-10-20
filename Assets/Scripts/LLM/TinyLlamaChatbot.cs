using System;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using System.Threading.Tasks;
using System.Collections.Generic;
using SIL.Machine.Tokenization.SentencePiece;

[DisallowMultipleComponent]
public class TinyLlamaChatbot : MonoBehaviour
{
    [Header("UI")]
    public InputField input;
    public Button sendBtn;
    public Text chatView;

    [Header("Files (StreamingAssets)")]
    public string onnxRelativePath = "LLM/model_q4f16.onnx";
    public string spmRelativePath = "LLM/tokenizer.model";
    public string specialTokensRelativePath = "LLM/special_tokens_map.json";

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

    private int _bosId = -1, _eosId = -1, _padId = 0;

    private Model _model;
    private IWorker _worker;
    private SilSpmTokenizer _tok;

    private readonly List<int> _context = new();

    void Awake()
    {
        _tok = new SilSpmTokenizer("LLM/tokenizer.model");

        try
        {
            var sp = Path.Combine(Application.streamingAssetsPath, specialTokensRelativePath);
            if (File.Exists(sp))
            {
                var json = JsonUtility.FromJson<SpecialTokensJson>(File.ReadAllText(sp).Replace("\n", ""));
                _bosId = TryGet(json.bos_token_id, -1);
                _eosId = TryGet(json.eos_token_id, -1);
                _padId = TryGet(json.pad_token_id, 0);
            }
        }
        catch { /*  */ }

        var onnxPath = Path.Combine(Application.streamingAssetsPath, onnxRelativePath);
        byte[] onnx = File.ReadAllBytes(onnxPath);
        _model = ModelLoader.Load(onnx);

        _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, _model);

        sendBtn.onClick.AddListener(OnSend);
        LogLine("?? Chatbot ready. ????? Send?");
    }

    void OnDestroy()
    {
        _worker?.Dispose();
        _model?.Dispose();
    }

    void OnSend()
    {
        var userText = input.text?.Trim();
        if (string.IsNullOrEmpty(userText)) return;
        input.text = "";

        LogLine($"?? {userText}");

        _ = GenerateAndShowAsync(userText);
    }

    async Task GenerateAndShowAsync(string userText)
    {
        var promptTokens = BuildPromptTokens(userText);

        var newTokens = await Task.Run(() => GenerateTokens(promptTokens, maxNewTokens));

        var reply = _tok.DecodeIds(newTokens.ToArray());
        LogLine($"?? {reply}");
    }

    List<int> BuildPromptTokens(string userText)
    {
        var tokens = _tok.EncodeIds(userText).ToList();

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
            var seqLen = workSeq.Count;
            var inputIdsArr = workSeq.ToArray();

            using var inputIds = new TensorInt(new TensorShape(1, seqLen), inputIdsArr);
            using var attnMask = new TensorInt(new TensorShape(1, seqLen), Enumerable.Repeat(1, seqLen).ToArray());

            var inputs = new Dictionary<string, Tensor>
            {
                { inputIdsName, inputIds },
                { attnMaskName, attnMask }
            };

            _worker.Execute(inputs);

            using var logits = _worker.PeekOutput(logitsName) as TensorFloat;
            var shape = logits.shape; // (1, seq, vocab)
            int vocab = shape[2];

            var lastLogits = new float[vocab];
            logits.ReadArray(lastLogits, startIndex: (seqLen - 1) * vocab, count: vocab);

            int nextId = SampleFromLogits(lastLogits, temperature, topK, topP);
            if (_eosId >= 0 && nextId == _eosId)
                break;

            newTokens.Add(nextId);
            workSeq.Add(nextId);
        }

        _context.AddRange(_tok.EncodeIds(" "));
        _context.AddRange(_tok.EncodeIds(string.Empty));
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
        for (int i = 0; i < scaled.Length; i++)
        {
            scaled[i] = (float)Math.Exp(scaled[i] - maxLogit);
            sum += scaled[i];
        }
        for (int i = 0; i < scaled.Length; i++) scaled[i] = (float)(scaled[i] / sum);

        var idx = Enumerable.Range(0, scaled.Length).ToArray();
        Array.Sort(scaled, idx);
        Array.Reverse(scaled); Array.Reverse(idx);
        int keep = (topK > 0) ? Math.Min(topK, scaled.Length) : scaled.Length;
        var probs = new List<(int id, float p)>(keep);
        for (int i = 0; i < keep; i++) probs.Add((idx[i], scaled[i]));

        if (topP > 0 && topP < 1.0f)
        {
            float c = 0; int cut = keep;
            for (int i = 0; i < keep; i++)
            {
                c += probs[i].p;
                if (c >= topP) { cut = i + 1; break; }
            }
            probs = probs.Take(cut).ToList();
        }

        float total = probs.Sum(x => x.p);
        float r = UnityEngine.Random.Range(0f, total);
        float acc = 0f;
        foreach (var (id, p) in probs)
        {
            acc += p;
            if (r <= acc) return id;
        }
        return probs[0].id;
    }

    static int TryGet(int v, int fallback) => v != 0 ? v : fallback;

    void LogLine(string s)
    {
        if (chatView) chatView.text += s + "\n";
        else Debug.Log(s);
    }

    [Serializable]
    class SpecialTokensJson
    {
        public int bos_token_id;
        public int eos_token_id;
        public int pad_token_id;
    }
}
