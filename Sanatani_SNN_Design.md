# Sanatani Spiking Neural Network (SSNN) Architecture

## Core Components Mapping

### 1. Kshan (The Discrete Moment of Time)
**Philosophy:** Time is a sequence of discrete moments (Kshan). Reality unfolds moment by moment.
**SNN Mapping:** `Timesteps (dt)`. SNNs are fundamentally temporal. Unlike standard ANNs which are static, SNNs process data across discrete timesteps. Every calculation (membrane potential update, spike generation) happens in a specific *Kshan*.

### 2. Man (The Fluctuating Mind)
**Philosophy:** Man (Mind) is the accumulator of sensory inputs (Indriyas). It is constantly fluctuating, gathering data, but indecisive.
**SNN Mapping:** `Membrane Potential Dynamics` and `Hidden Layers (Sensory Processing)`. The neuron's membrane potential goes up and down based on inputs, much like the fluctuating mind. It gathers excitatory and inhibitory signals but doesn't make the final decision until a threshold is reached.

### 3. Buddhi (The Decisive Intellect)
**Philosophy:** Buddhi is the intellect that discriminates (Viveka) and makes a firm decision.
**SNN Mapping:** `The Spike Threshold` and `Output/Decision Layers`. When the fluctuating mind (membrane potential) reaches absolute clarity (the Threshold), Buddhi takes over and fires a Spike (Action Potential). This spike is the final, binary decision (1 = Yes, 0 = No) cutting through the noise.

### 4. Karma & Samskara (Action and Memory/Impression)
**Philosophy:** Every action (Karma) leaves an impression (Samskara). Repeated actions strengthen habits.
**SNN Mapping:** `STDP (Spike-Timing-Dependent Plasticity)`. 
- **Karma (Action):** The firing of a spike.
- **Samskara (Impression/Habit):** The updating of Synaptic Weights. If Neuron A repeatedly causes Neuron B to fire, their bond strengthens (Good Karma/Potentiation). If they are out of sync, the bond weakens (Depression).

---
## Proposed Architecture Flow:

[Input Environment] 
   | (Sensory Data)
   V
[Man (Mind / Encoding Layer)] --> Evaluates inputs, membrane potential fluctuates.
   | (Integration over Kshan/Time)
   V
[Buddhi (Intellect / Threshold)] --> Makes the decision (Spikes) when clarity is reached.
   |
   V
[Karma (STDP Learning)] --> Spikes trigger weight updates (Samskaras), modifying future behavior.

---

## Technical Execution Plan (Build Phases)

### Phase 1: The Core Neuron (Man, Buddhi, Kshan & Triguna)
*   **Goal:** Create a foundational `ManBuddhiNeuron` using PyTorch.
*   **Parameters:** 
    *   `Tamas` (Leak factor/Inertia - decays potential over time)
    *   `Rajas` (Input current/Activity - increases potential)
    *   `Sattva` (Threshold - balance point where Buddhi generates a spike)
*   **Process:** Iterate over `Kshan` (time steps), allowing 'Man' to fluctuate and 'Buddhi' to decide (spike).

### Phase 2: Action & Impression (Karma & Samskara - STDP)
*   **Goal:** Implement Unsupervised Learning (STDP).
*   **Rule:** If Pre-synaptic spike precedes Post-synaptic spike within a short window, weight (Samskara) increases (Good Karma). Otherwise, decreases.

### Phase 3: The Koshas Architecture (Layering)
*   **Goal:** Assemble neurons into a layered `PanchakoshaNetwork`.
*   **Layers:** Physical Input (`Annamaya`), Encoding (`Pranamaya`), Mental Processing (`Manomaya`), Intellect Features (`Vijnanamaya`), Classification Output (`Anandamaya`).
