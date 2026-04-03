import torch
import torchvision
import torchvision.transforms as transforms
import time
import json
from collections import defaultdict

from ssnn_indriyas import NetraIndriya
from ssnn_network import PanchkoshaNetwork

def train_and_save():
    print("=============================================")
    print("  SSNN - TRAINING AND EXPORTING SNN MODEL    ")
    print("=============================================\n")

    transform = transforms.Compose([transforms.ToTensor()])
    print("[1] Loading MNIST Dataset...")
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Extract subset of digits [0, 1, 2, 3, 4] for a 5-digit classification model
    samples, labels = [], []
    for img, label in trainset:
        if label in [0, 1, 2, 3, 4]:
            samples.append(img.flatten())
            labels.append(label)
        if len(samples) >= 2000:
            break
            
    print(f"    Train Samples collected: {len(samples)}")

    # Initialize SNN with 15 Buddhi Neurons
    print("\n[2] Waking up Panchkosha Network (15 Buddhi Neurons)...")
    ssnn = PanchkoshaNetwork(input_size=784, hidden_size=15)
    
    torch.manual_seed(108)
    ssnn.samskaras.data = torch.rand(784, 15) * 0.8
    
    eye = NetraIndriya(num_kshan=15) 
    
    # ------------------ PHASE 1: UNSUPERVISED LEARNING ------------------
    epochs = 4
    print(f"\n[3] Phase 1 - Unsupervised Karma Accumulation (STDP) for {epochs} Epochs...")
    start = time.time()
    for epoch in range(epochs):
        print(f"  --- Epoch {epoch+1}/{epochs} ---")
        for idx, img in enumerate(samples):
            spikes = eye.forward_image(img).unsqueeze(1) # [Time, 1, 784]
            ssnn.experience_life(spikes) # STDP learning happens here
            
            if (idx+1) % 500 == 0:
                print(f"    -> Experienced {idx+1} images...")
            
    print(f"    Training done in {time.time() - start:.2f}s.")

    # ------------------ PHASE 2: LABEL ASSIGNMENT ------------------
    print("\n[4] Phase 2 - Naming the Buddhis (Label Assignment)...")
    neuron_label_counts = {i: defaultdict(int) for i in range(15)}

    for img, label in zip(samples, labels):
        spikes = eye.forward_image(img).unsqueeze(1)
        with torch.no_grad():
            buddhi_spikes = ssnn(spikes).squeeze(1) # Shape: [Time, 15]
            total_spikes_per_neuron = buddhi_spikes.sum(dim=0)
            best_neuron = torch.argmax(total_spikes_per_neuron).item()
            
            neuron_label_counts[best_neuron][label] += 1

    neuron_to_label = {}
    for neuron_id, counts in neuron_label_counts.items():
        if counts:
            best_label = max(counts, key=counts.get)
            neuron_to_label[neuron_id] = best_label
            print(f"    Buddhi {neuron_id} mostly fired for Digit: {best_label} (Counts: {dict(counts)})")
        else:
            neuron_to_label[neuron_id] = -1 
            print(f"    Buddhi {neuron_id} remained silent.")

    # ------------------ PHASE 3: EXPORT MODEL ------------------
    print("\n[5] Phase 3 - Saving Samskaras (Weights) & Labels...")
    torch.save(ssnn.samskaras.data, "ssnn_samskaras.pt")
    with open("neuron_to_label.json", "w") as f:
        json.dump(neuron_to_label, f)
        
    print("\n\u2728 Model Successfully Exported \u2728")
    print("Files saved: 'ssnn_samskaras.pt' and 'neuron_to_label.json'")

if __name__ == "__main__":
    train_and_save()