import gradio as gr
import torch
import json
import torchvision.transforms as transforms
import numpy as np

# Import our custom Sanatani SNN Architecture
from ssnn_indriyas import NetraIndriya
from ssnn_network import PanchkoshaNetwork

print("Loading the Preserved Samskaras (Trained Model Weights)...")

# 1. Initialize Network
ssnn = PanchkoshaNetwork(input_size=784, hidden_size=15)
eye = NetraIndriya(num_kshan=15)

# 2. Load Weights and Labels 
try:
    ssnn.samskaras.data = torch.load("ssnn_samskaras.pt")
    with open("neuron_to_label.json", "r") as f:
        neuron_to_label = json.load(f)
        # Convert dictionary keys from string to int
        neuron_to_label = {int(k): v for k, v in neuron_to_label.items()}
    print("\u2705 Model Loaded Successfully!")
except FileNotFoundError:
    print("\u274c Error: Could not find 'ssnn_samskaras.pt' or 'neuron_to_label.json'. Run 'train_and_save.py' first.")
    exit(1)


def predict(image_dict):
    """
    Takes an image drawn on Gradio canvas, feeds it to NetraIndriya,
    runs the forward pass via PanchkoshaNetwork, and reads Buddhi decisions.
    image_dict contains 'background' and 'layers'. The drawn digit is usually in 'layers'[0] or 'composite'.
    Gradio sketchpad gives us an image dict or numpy array. We'll handle both.
    """
    if image_dict is None:
        return "Please draw a digit (0 to 4)", {}
    
    # Gradio passes an image dict for 'sketchpad'
    image = image_dict["composite"]
    
    # Convert numpy array (RGBA) to grayscale Grayscale -> (H, W)
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Invert background (Black background, white digit)
        # Gradio sketchpad is usually transparent drawing black, 
        # let's just grab the alpha channel as brightness
        grayscale = image[:, :, 3] 
    else:
        grayscale = image  # Assuming already 2D

    # Resize to 28x28 like MNIST dataset
    from PIL import Image
    pil_image = Image.fromarray(grayscale).resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to Tensor and Normalize
    tensor_img = transforms.ToTensor()(pil_image).flatten()
    
    if tensor_img.max() == 0:
        return "No drawing detected.", {}
        
    # Scale it so it fires spikes properly
    tensor_img = tensor_img / tensor_img.max()

    # Pass through SNN
    with torch.no_grad():
        # Eye creates spikes
        spikes = eye.forward_image(tensor_img).unsqueeze(1) # [Time, 1, 784]
        
        # Buddhi processes through Samskaras
        buddhi_spikes = ssnn(spikes).squeeze(1) # [Time, 15]
        
        # Total decision per neuron
        total_spikes_per_neuron = buddhi_spikes.sum(dim=0)
        
    # Map back to Labels
    spike_counts = {}
    for i in range(15):
        spikes = total_spikes_per_neuron[i].item()
        label = neuron_to_label.get(i, -1)
        if label != -1: # Only track active neurons
            name = f"Buddhi {i} (Learned '{label}')"
            spike_counts[name] = spikes

    # Predict final answer (which digit class got the most sum of spikes)
    final_scores = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for i in range(15):
        label = neuron_to_label.get(i, -1)
        if label != -1:
            final_scores[label] += total_spikes_per_neuron[i].item()
            
    highest_score = max(final_scores.values())
    if highest_score == 0:
         final_prediction = "Model is confused / Indecisive."
    else:
         best_digit = max(final_scores, key=final_scores.get)
         final_prediction = f"Digit \u27a1\ufe0f **{best_digit}**"

    return final_prediction, final_scores

# ----------------- Gradio UI -----------------
with gr.Blocks() as app:
    gr.Markdown("# 🕉️ Sanatani Spiking Neural Network Web App")
    gr.Markdown("Draw a digit between **0 to 4** on the canvas below. The image will be converted to *Prana Spikes*, and the *Buddhi* neurons will cast their final votes based on their *Samskaras* (Unsupervised learning).")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Draw Here")
            # Image painter (Sketchpad) expects user to draw
            canvas = gr.Sketchpad(canvas_size=(280, 280), type="numpy", brush=gr.Brush(default_size=20, colors=["#000000"]))
            submit_btn = gr.Button("Consult Buddhi (Predict)", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Prediction (Decisive Intellect)")
            result_text = gr.Markdown("Waiting for input...")
            gr.Markdown("### Spike Activities (Label Mapping)")
            result_plot = gr.Label(num_top_classes=5)
            
    submit_btn.click(fn=predict, inputs=canvas, outputs=[result_text, result_plot])
    gr.Markdown("--- \n *Built purely via Karma STDP and Temporal Processing (No Backpropagation). Models trained via 'tapasya' and 'ahamkara'.*")

if __name__ == "__main__":
    # Launch local server
    app.launch(share=False, theme=gr.themes.Monochrome())
