An integration Unity project integrated the smallest local language model (LM) I can find (and also supported by Unity inference)
The LM download URL: https://huggingface.co/onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX/tree/main/onnx

1. download the model\_fp16.onnx and model\_fp16.onnx\_data
2. create a new Unity project, has to be Unity 6.2 or newer
3. clone the repo from GitHub, replace the files in the new project folder
4. place model\_fp16.onnx and model\_fp16.onnx\_data in the Assets/LLM folder
5. open Unity and wait for compiling
6. Assign the model to Canvas/conversation's script in the Unity inspector
7. Run it, should be good to go
