import torch
import torch.nn as nn
import os

from Models import ResNet1D  # Adjust import paths as per your code structure
# Make sure ModelBase is defined similarly as in your provided code snippet

class ModelBase(nn.Module):
    """
    Encoder for classification based on ResNet1D.
    """
    def __init__(self, dim=128):
        super(ModelBase, self).__init__()
        self.net = ResNet1D.resnet18(norm_layer=None)  # Encoder using ResNet18 architecture
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, dim)

    def forward(self, x):
        x = self.net(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def load_pretrained_encoder(model, pretrain_path):
    """
    Load pre-trained encoder weights into the model.
    This replicates what was done in the fine-tuning code.
    """
    checkpoint = torch.load(pretrain_path, map_location='cpu')

    # Extract encoder weights from 'encoderT'
    encoder_state_dict = {}
    for k in list(checkpoint.keys()):
        if k.startswith('encoderT.'):
            new_key = k.replace('encoderT.', '')
            encoder_state_dict[new_key] = checkpoint[k]

    # Remove 'fc' layer weights if they exist, to avoid mismatch
    encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if not k.startswith('fc.')}

    missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)
    print("After loading pre-trained encoder:")
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # Initialize fc weights
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    return model

def load_finetuned_weights(model, finetune_path):
    """
    Load the fine-tuned weights (including fc layer) from best_model.pth.
    This will overwrite the initial fc initialization with the fine-tuned fc weights.
    """
    fine_checkpoint = torch.load(finetune_path, map_location='cpu')
    # This should contain both encoder and fc weights as saved after fine-tuning.
    model.load_state_dict(fine_checkpoint, strict=True)
    print("Fine-tuned weights loaded successfully.")
    return model

def get_final_model(pretrained_path, finetuned_path, num_classes, device='cpu'):
    """
    Returns a model with pre-trained encoder loaded, then fine-tuned weights loaded.
    Args:
        pretrained_path (str): Path to the initial pre-trained encoder checkpoint (e.g. TFPred_checkpoint.pth).
        finetuned_path (str): Path to the fine-tuned weights (e.g. best_model.pth).
        num_classes (int): Number of classes for the final classification.
        device (str): 'cpu' or 'cuda'.

    Returns:
        nn.Module: The fully loaded model ready for inference.
    """
    model = ModelBase(dim=num_classes).to(device)
    model = load_pretrained_encoder(model, pretrained_path)
    model = load_finetuned_weights(model, finetuned_path)
    model.eval()  # Set to evaluation mode
    return model

# Example usage in demo.py:
if __name__ == "__main__":
    # Paths to your checkpoints
    pretrained_checkpoint = "./History/TFPred_checkpoint.pth"  # original pre-training checkpoint
    finetuned_checkpoint = "./best_model.pth"  # fine-tuned checkpoint from your training script

    # Suppose you have 4 classes
    num_classes = 4

    # Load the final model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_final_model(pretrained_checkpoint, finetuned_checkpoint, num_classes, device=device)

    # Now 'model' has both the pre-trained encoder and fine-tuned fc weights.
    # You can use 'model' in your Gradio demo or inference functions.

import torch
import numpy as np
import matplotlib.pyplot as plt

# Suppose these were your classes during fine-tuning:
classes = ["bearing fault", "imbalance", "misalignment", "normal"]

def preprocess_input(input_str, length=1024):
    values = input_str.strip().split(',')
    if len(values) != length:
        return None, f"Invalid input length: expected {length} values, got {len(values)}."
    try:
        arr = np.array([float(v.strip()) for v in values], dtype=np.float32)
    except ValueError:
        return None, "Invalid values. Ensure all inputs are numeric."
    arr = arr.reshape(1, 1, length)
    tensor = torch.tensor(arr)
    return tensor, None

def predict_fault(model, input_str, sampling_rate, device='cpu'):
    tensor, error = preprocess_input(input_str)
    if error:
        # Return error message and None for the figure
        return error, None

    tensor = tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = classes[pred_idx]
        confidence = probs[pred_idx].item()

    prediction_text = f"Predicted Fault: {pred_class} (Confidence: {confidence*100:.2f}%)"

    # Extract the original waveform array (1024 samples)
    arr = tensor.cpu().numpy().flatten()  # shape was (1,1,1024)

    # Compute FFT with the given sampling rate
    fft_data = np.fft.fft(arr)
    n = len(arr)
    # Frequency bins using sampling rate fs:
    # np.fft.fftfreq(n, d=1/fs)
    fs = sampling_rate if sampling_rate > 0 else 1.0  # ensure fs > 0 to avoid division by zero
    freq = np.fft.fftfreq(n, d=1/fs)

    # Only consider the positive half of the spectrum for plotting
    half = n // 2
    freq_half = freq[:half]
    fft_mag = np.abs(fft_data[:half])

    # Plot FFT
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(freq_half, fft_mag)
    ax.set_title('FFT of Input Waveform')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

    return prediction_text, fig
import gradio as gr

def gradio_inference(input_str, sampling_rate):
    # input_str: waveform data string
    # sampling_rate: numeric value for fs
    result_text, fft_figure = predict_fault(model, input_str, sampling_rate, device=device)
    return result_text, fft_figure

interface = gr.Interface(
    fn=gradio_inference,
    inputs=[
        gr.Textbox(
            lines=10,
            placeholder="Enter 1024 comma-separated float values representing the waveform.",
            label="Waveform Data Input"
        ),
        gr.Number(
            value=1000,  # default sampling rate
            label="Sampling Rate (Hz)",
            precision=0
        )
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Plot(label="FFT Plot")
    ],
    title="Rotatory Machinery Fault Classifier",
    description=(
        "This is a live demonstation of the Rotatory Machinery Fault Classifier model."
        "Insert a 1024 length comma seperated waveform data and then click 'Submit' to classify the fault type. "
        "The model will also compute and display the FFT magnitude spectrum of the input signal using the given sampling rate."
    ),
    theme="soft"
)

if __name__ == "__main__":
    interface.launch(share = True)
