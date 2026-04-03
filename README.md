# Sanatani Spiking Neural Network (SSNN)

An innovative, purely temporal **Spiking Neural Network (SNN)** mapped entirely to ancient **Sanatani / Vedic philosophy**. This architecture bypasses traditional backpropagation and standard deep learning mechanics (like backprop, static neurons, loss functions) in favor of **Unsupervised Hebbian Learning (Karma STDP)**, **Lateral Inhibition (Ahamkara)**, and **Adaptive Thresholding (Tapasya)**.

## 🌟 Philosophy to Technology Mapping

| Sanatani Concept | SNN Technical Equivalent | Description |
| :--- | :--- | :--- |
| **Kshan (Moment/Time)** | `dt / Timesteps` | Networks process dynamically over discrete time steps. |
| **Man (Mind)** | `Membrane Potential` | Constantly fluctuates (Tamas/Decay vs Rajas/Input current). |
| **Buddhi (Intellect)** | `Spiking Threshold` | The decisive intellect that cuts through noise and fires a binary spike (1). |
| **Karma & Samskara** | `STDP (Hebbian Learning)` | Actions (Spikes) leave Impressions (Weights/Samskaras). Good timing strengthens the bond (Punya), bad timing weakens it (Paap). |
| **Ahamkara (Ego)** | `WTA / Lateral Inhibition` | "Winner-takes-all." When one Buddhi decides, it fiercely suppresses the others to claim the pattern. |
| **Tapasya (Asceticism)** | `Adaptive Threshold` | The more a dominant neuron fires, the harder its threshold becomes, forcing it to allow silent neurons a chance to learn (Homeostasis). |
| **Niyama (Discipline)** | `Weight Normalization` | Samskaras (Weights) cannot grow infinitely. They are balanced so the total energy entering a neuron is constrained. |

## 🧬 Architecture Overview

*   **Panchkosha Architecture**: Layers structured loosely around the 5 Koshas.
    *   **Indriyas (Sensory Organs)**: e.g., `NetraIndriya` (Eye) converts pixel intensities into Poisson-distributed spike trains.
    *   **Vijnanamaya Kosha (Triguna Neuron)**: The core dynamic spiking module where *Man* fluctuates and *Buddhi* decides.
*   **Karma STDP Layer**: The network learns features entirely unsupervised. 

## 🚀 Quick Start

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Visualizing Unsupervised Receptive Fields (`train_mnist.py`)**:
   Train the network purely via time and karma without labels. It will evolve its own weights (Samskaras).
   ```bash
   python train_mnist.py
   ```
   *Generates an image (`mnist_samskaras.png`) visualizing the "features" each Buddhi neuron has learned.*

3. **Evaluate the Cluster Accuracy (`evaluate_mnist.py`)**:
   Measure how accurately the unsupervised clustering performs on unseen MNIST data.
   ```bash
   python evaluate_mnist.py
   ```

## 📂 Project Structure

*   `ssnn_core.py`: The `TrigunaNeuron` containing logic for *Tamas* (decay), *Sattva* (threshold), *Ahamkara* (WTA), and *Tapasya* (Adaptive Threshold).
*   `ssnn_indriyas.py`: Sensors like `NetraIndriya` (Vision) and `KarnaIndriya` (Audio) that translate physical world continuous tensors into temporal discrete spikes over *Kshan*.
*   `ssnn_network.py`: The full `PanchkoshaNetwork` and `KarmaSTDP` learning rule.
*   `Sanatani_SNN_Design.md`: Detailed original architectural concept and design document.

---
*Created with intent to explore biological plausibility mapped with ancient Indian metaphysical concepts.*
