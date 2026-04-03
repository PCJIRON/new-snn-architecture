import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

from ssnn_core import TrigunaNeuron
from ssnn_indriyas import NetraIndriya
from ssnn_network import PanchkoshaNetwork

def main():
    print("=============================================")
    print("      SSNN - EXPERIENCING MNIST DATASET      ")
    print("=============================================\n")

    print("[1] Loading MNIST Dataset from Torchvision...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Download dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # We will use just a subset of 0s, 1s, and 2s to make the demo fast
    # Real unsupervised learning takes thousands of epochs, but our Ego (Ahamkara) 
    # network learns extremely fast in just one pass.
    samples = []
    labels = []
    
    for i in range(len(trainset)):
        img, label = trainset[i]
        if label in [0, 1, 2]:
            samples.append(img)
            labels.append(label)
        if len(samples) >= 300: # 300 images total for quick training
            break
            
    print(f"Captured {len(samples)} images of digits 0, 1, and 2.")

    # Initialize Sanatani SNN
    print("\n[2] Initializing Panchkosha Network...")
    print("    Input: 784 pixels (28x28)")
    print("    Output: 6 Buddhi Neurons (Hoping they differentiate different digits/strokes)")
    
    # 6 Buddhi neurons with slightly lower initial random noise
    ssnn = PanchkoshaNetwork(input_size=784, hidden_size=6)
    # Initialize with stronger random variations to break symmetry instantly
    ssnn.samskaras.data = torch.rand(784, 6) * 0.5
    
    # Kshan = 15 moments of observation per image
    eye = NetraIndriya(num_kshan=15) 
    
    # Training Loop
    print("\n[3] Living life and accumulating Karma (Training on 300 images)...")
    start = time.time()
    
    for idx, img in enumerate(samples):
        # img is [1, 28, 28] -> flat [784]
        flat_img = img.flatten()
        
        # Add visual stimulus -> Prana Spikes => [num_kshan, 1, 784]
        # Netra converts pixel intensity to Poisson spikes over time
        spikes = eye.forward_image(flat_img).unsqueeze(1)
        
        # Network experiences the image, fluctuates Man, Buddhi spikes, and updates Karma
        ssnn.experience_life(spikes)
        
        if (idx+1) % 50 == 0:
            print(f"  -> Experienced {idx+1} images...")
            
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds.")
    
    # Visualize the Samskaras (Weights) learned
    print("\n[4] Visualizing the learned Samskaras (Habits/Weights)...")
    print("Saving to 'mnist_samskaras.png'")
    weights = ssnn.samskaras.data.clone().cpu() # Shape: [784, 6]
    
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for i in range(6):
        # Reshape [784] to [28, 28] to view as an image
        img_w = weights[:, i].view(28, 28).numpy()
        axes[i].imshow(img_w, cmap='hot')
        axes[i].set_title(f"Buddhi {i}")
        axes[i].axis('off')
        
    plt.suptitle("Sanatani SNN - Unsupervised Learned MNIST Features (Samskaras)", fontsize=16)
    plt.tight_layout()
    plt.savefig('mnist_samskaras.png')
    print("Done! Open 'mnist_samskaras.png' in VS Code to see what the Neurons learned! \u2728")

if __name__ == "__main__":
    main()