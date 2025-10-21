using UnityEditor;
using UnityEngine;
using Unity.InferenceEngine;

public static class ExportSentis
{
    [MenuItem("LLM/Export .sentis (no quant)")]
    public static void Export()
    {
        var sel = Selection.activeObject as ModelAsset;
        if (sel == null) { Debug.LogError("Select a ModelAsset (.onnx) in Project view."); return; }

        System.IO.Directory.CreateDirectory(Application.streamingAssetsPath);
        var outPath = System.IO.Path.Combine(Application.streamingAssetsPath, sel.name + ".sentis");

        ModelWriter.Save(outPath, sel);
        Debug.Log($"Saved: {outPath}");
        AssetDatabase.Refresh();
    }
}
