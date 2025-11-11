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

    [Header("Model I/O Names (match Inspector exactly)")]
    [SerializeField] private string inputIdsName = "input_ids";
    [SerializeField] private string attnMaskName = "attention_mask";
    [SerializeField] private string logitsName = "logits";

    private int _bosId, _eosId, _padId;
    private Model _model;
    private Worker _worker;
    private SilSpmTokenizer _tok;
    private readonly List<int> _context = new();

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    static void BootProbe() => Debug.Log("[LLM] BootProbe: BeforeSceneLoad");

    void OnEnable() => Debug.Log("[LLM] OnEnable");
    void Start() => Debug.Log("[LLM] Start()");

    void Awake()
    {
        Debug.Log("[LLM] Awake: begin");

        if (sendBtn == null) Debug.LogError("[LLM] sendBtn is NOT assigned in Inspector");
        if (input == null) Debug.LogError("[LLM] input (TMP_InputField) is NOT assigned in Inspector");
        if (chatView == null) Debug.LogWarning("[LLM] chatView is NOT assigned, logs will go Console only");

        try
        {
            // 1) tokenizer
            _tok = new SilSpmTokenizer();
            _bosId = _tok.BosId; _eosId = _tok.EosId; _padId = _tok.PadId;
            Debug.Log($"[LLM] Tokenizer ready. BOS={_bosId} EOS={_eosId} PAD={_padId}");

            // 2) load .sentis
            var baseDir = Application.streamingAssetsPath;
            var path = Path.Combine(baseDir, sentisFileName);
            Debug.Log($"[LLM] streamingAssetsPath = {baseDir}");
            Debug.Log($"[LLM] expecting .sentis   = {sentisFileName}");
            foreach (var f in Directory.GetFiles(baseDir, "*.sentis")) Debug.Log("[LLM] found .sentis = " + f);

            if (!File.Exists(path))
                throw new FileNotFoundException($".sentis model not found: {path}");

            _model = ModelLoader.Load(path);
            _worker = new Worker(_model, BackendType.CPU); //use CPU first, then switch to GPUCompute
            Debug.Log("[LLM] Model loaded & Worker created (CPU)");

            // 3) bind UI & callback
            if (sendBtn != null)
                sendBtn.onClick.AddListener(OnSend);
            LogLine("Chatbot ready. Type and click Send.");
        }
        catch (Exception ex)
        {
            Debug.LogError("[LLM] Awake failed: " + ex);
        }

        Debug.Log("[LLM] Awake: end");
    }

    void OnDestroy()
    {
        Debug.Log("[LLM] OnDestroy");
        _worker?.Dispose();
        _tok?.Dispose();
    }

    void LogLine(string s)
    {
        Debug.Log("[LLM] " + s);
        if (chatView) chatView.text += s + "\n";
    }

    void OnSend()
    {
        Debug.Log("[LLM] OnSend clicked");
        var userText = input != null ? input.text?.Trim() : null;
        if (string.IsNullOrEmpty(userText)) { LogLine("(empty input)"); return; }

        // fake ping command for testing responsiveness
        if (userText.Equals("/ping", StringComparison.OrdinalIgnoreCase))
        {
            LogLine("/ping");
            LogLine("pong");
            if (input) input.text = "";
            return;
        }

        LogLine(userText);
        if (input) input.text = "";
        _ = GenerateAndShowAsync(userText);
    }

    async Task GenerateAndShowAsync(string userText)
    {
        try
        {
            var promptTokens = BuildPromptTokens(userText);
            var newTokens = await Task.Run(() => GenerateTokens(promptTokens, maxNewTokens));
            var reply = _tok.Decode(newTokens.ToArray());
            LogLine(reply);
        }
        catch (Exception ex)
        {
            Debug.LogError("[LLM] GenerateAndShowAsync failed: " + ex);
        }
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

            using var inputIdsTensor = new Tensor<long>(new TensorShape(1, seqLen), idsArr64);
            _worker.SetInput(inputIdsName, inputIdsTensor);

            if (!string.IsNullOrEmpty(attnMaskName))
            {
                using var maskTensor = new Tensor<long>(new TensorShape(1, seqLen), maskArr64);
                _worker.SetInput(attnMaskName, maskTensor);
            }

            _worker.SetInputShapeDimension(inputIdsName, 1, seqLen);
            if (!string.IsNullOrEmpty(attnMaskName))
                _worker.SetInputShapeDimension(attnMaskName, 1, seqLen);

            _worker.Schedule();

            var logitsTensor = _worker.PeekOutput(logitsName) as Tensor<float>
                               ?? _worker.PeekOutput() as Tensor<float>;

            if (logitsTensor == null)
            {
                Debug.LogError($"[LLM] Output '{logitsName}' not found & default null.");
                return newTokens;
            }

            logitsTensor.CompleteAllPendingOperations();
            var raw = logitsTensor.DownloadToArray();

            var s = logitsTensor.shape;
            var sArr = s.ToArray();
            Debug.Log($"[LLM] step={step} logits shape=({string.Join(",", sArr)}) len={raw.Length}");

            int rank = s.rank;
            int vocab = s[-1];
            int timeIndex = (rank >= 3 && s[1] > 1) ? (seqLen - 1) : 0;

            int start = timeIndex * vocab;
            if (start + vocab > raw.Length)
            {
                Debug.LogError($"[LLM] bad slice: start={start}, vocab={vocab}, rawLen={raw.Length}");
                return newTokens;
            }

            var lastLogits = new float[vocab];
            Array.Copy(raw, start, lastLogits, 0, vocab);

            if (step == 0 && _eosId >= 0 && _eosId < lastLogits.Length)
                lastLogits[_eosId] = float.NegativeInfinity;

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
}
