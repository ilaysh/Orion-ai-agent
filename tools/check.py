from torchinfo import summary
import torch, onnxruntime as ort

sess = ort.InferenceSession("models/orion_wake.onnx")
print("Inputs:", [i.name for i in sess.get_inputs()])
print("Outputs:", [o.name for o in sess.get_outputs()])
print("Expected shape:", sess.get_inputs()[0].shape)