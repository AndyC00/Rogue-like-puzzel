An integration Unity project integrated the smallest local language model (LM) I can find (and also supported by Unity inference)
The LM download utl: https://huggingface.co/onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX/tree/main/onnx

1. download the model_fp16.onnx and model_fp16.onnx_data
2. place them in the Assets/LLM folder
3. open Unity and wait for the compile
4. Assign the model to Canvas/conversation's model in the Unity inspector
5. Run it, should be good to go
