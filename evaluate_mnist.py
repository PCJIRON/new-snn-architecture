import torch
import torchvision
import torchvision.transforms as transforms
import time
from collections import defaultdict

from ssnn_indriyas import NetraIndriya
from ssnn_network import PanchkoshaNetwork

def main():
    print("=============================================")
    print("  SSNN - UNSUPERVISED ACCURACY EVALUATION    ")
    print("=============================================\n")

    transform = transforms.Compose([transforms.ToTensor()])
    print("[1] Loading MNIST Dataset (Train & Test)...")
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Extract subset of digits [0, 1, 2]
    def get_subset(dataset, num_samples):
        samples, labels = [], []
        for img, label in dataset:
            if label in [0, 1, 2]:
                samples.append(img.flatten())
                labels.append(label)
            if len(samples) >= num_samples:
                break
        return samples, labels

    train_samples, train_labels = get_subset(trainset, 1500)  # Increased from 600 to 1500
    test_samples, test_labels = get_subset(testset, 400)     # Increased from 200 to 400
    
    print(f"    Train Samples: {len(train_samples)}")
    print(f"    Test Samples:  {len(test_samples)}")

    # Initialize SNN with 10 Buddhi Neurons (redundancy is good for unsupervised learning)
    print("\n[2] Waking up Panchkosha Network (10 Buddhi Neurons)...")
    ssnn = PanchkoshaNetwork(input_size=784, hidden_size=10)
    
    # Introduce larger initial variance so all neurons don't start the same 
    # and they fight more fairly during Ahamkara (WTA).
    torch.manual_seed(108)
    ssnn.samskaras.data = torch.rand(784, 10) * 0.8
    
    # Kshan = 15 moments of observation per image
    eye = NetraIndriya(num_kshan=15) 
    
    # ------------------ PHASE 1: UNSUPERVISED LEARNING ------------------
    epochs = 3
    print(f"\n[3] Phase 1 - Unsupervised Karma Accumulation (STDP) for {epochs} Epochs...")
    start = time.time()
    for epoch in range(epochs):
        print(f"  --- Epoch {epoch+1}/{epochs} ---")
        for idx, img in enumerate(train_samples):
            spikes = eye.forward_image(img).unsqueeze(1) # [Time, 1, 784]
            ssnn.experience_life(spikes) # STDP learning happens here
            
            if (idx+1) % 500 == 0:
                print(f"    -> Experienced {idx+1} images...")
            
    print(f"    Training done in {time.time() - start:.2f}s.")

    # ------------------ PHASE 2: LABEL ASSIGNMENT ------------------
    # Since it's unsupervised, the network doesn't know what a '0', '1', or '2' is.
    # It just knows "Pattern A", "Pattern B". We need to map which neuron learned which digit.
    print("\n[4] Phase 2 - Naming the Buddhis (Label Assignment)...")
    neuron_label_counts = {i: defaultdict(int) for i in range(10)}

    for img, label in zip(train_samples, train_labels):
        spikes = eye.forward_image(img).unsqueeze(1)
        # Pass data WITHOUT 'experience_life' to prevent STDP learning (Forward pass only)
        with torch.no_grad():
            buddhi_spikes = ssnn(spikes).squeeze(1) # Shape: [Time, 10]
            # Find which neuron spiked the most
            total_spikes_per_neuron = buddhi_spikes.sum(dim=0)
            best_neuron = torch.argmax(total_spikes_per_neuron).item()
            
            neuron_label_counts[best_neuron][label] += 1

    # Map each neuron to the label it fired most for
    neuron_to_label = {}
    for neuron_id, counts in neuron_label_counts.items():
        if counts:
            best_label = max(counts, key=counts.get)
            neuron_to_label[neuron_id] = best_label
            print(f"    Buddhi {neuron_id} mostly fired for Digit: {best_label} (Counts: {dict(counts)})")
        else:
            neuron_to_label[neuron_id] = -1 # Neuron didn't learn anything useful
            print(f"    Buddhi {neuron_id} remained silent/inactive.")

    # ------------------ PHASE 3: TESTING & ACCURACY ------------------
    print("\n[5] Phase 3 - The Final Test (Accuracy Evaluation)...")
    correct = 0
    total = len(test_samples)
    
    for img, label in zip(test_samples, test_labels):
        spikes = eye.forward_image(img).unsqueeze(1)
        with torch.no_grad():
            buddhi_spikes = ssnn(spikes).squeeze(1)
            total_spikes = buddhi_spikes.sum(dim=0)
            
            predicted_neuron = torch.argmax(total_spikes).item()
            predicted_label = neuron_to_label.get(predicted_neuron, -1)
            
            if predicted_label == label:
                correct += 1

    accuracy = (correct / total) * 100
    print("\n=============================================")
    print(f" \u2728 FINAL UNSUPERVISED ACCURACY: {accuracy:.2f}% \u2728")
    if accuracy > 50:
         print(" Excellent! It easily beats random guessing (which would be ~33%).")
    print("=============================================")

if __name__ == "__main__":
    main()