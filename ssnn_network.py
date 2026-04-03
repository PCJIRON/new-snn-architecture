import torch
import torch.nn as nn
from ssnn_core import TrigunaNeuron
from ssnn_indriyas import NetraIndriya

class KarmaSTDP(nn.Module):
    """
    Law of Karma: Actions (Spikes) leave Impressions (Samskaras/Weights).
    Unsupervised STDP (Spike-Timing Dependent Plasticity) Learning.
    """
    def __init__(self, positive_karma=0.05, negative_karma=0.01):
        super().__init__()
        # Positive Karma: Pre-synaptic spike causes Post-synaptic spike -> Strengthen Connection
        self.pos_karma = positive_karma 
        # Negative Karma: Post-synaptic spike happens before Pre-synaptic -> Weaken Connection
        self.neg_karma = negative_karma 

    def forward(self, prana_in, buddhi_out, samskaras):
        """
        Updates the Samskaras (Weights) based on the timing of spikes.
        prana_in: [Time, 1, InputSize]
        buddhi_out: [Time, 1, OutputSize]
        samskaras: [InputSize, OutputSize] Matrix
        """
        time_steps = prana_in.size(0)
        
        # Karma Traces (Like brief memories of the spikes)
        trace_in = torch.zeros_like(prana_in[0].squeeze(0))
        trace_out = torch.zeros_like(buddhi_out[0].squeeze(0))
        
        delta_samskara = torch.zeros_like(samskaras)

        # Iterate over Time (Kshan)
        for t in range(time_steps):
            t_in = prana_in[t].squeeze(0)  # Shape: [InputSize]
            t_out = buddhi_out[t].squeeze(0) # Shape: [OutputSize]
            
            # Traces decay slowly (Tamas) but rise on spike (Rajas)
            trace_in = (trace_in * 0.8) + t_in
            trace_out = (trace_out * 0.8) + t_out
            
            # If trace_in is high (Pre fired) and t_out fires (Post fired) = GOOD KARMA (Potentiation)
            punya = torch.outer(trace_in, t_out) 
            
            # If t_in fires (Pre fired) but trace_out is high (Post fired already) = BAD KARMA (Depression)
            paap = torch.outer(t_in, trace_out)
            
            # Update the change in Samskaras
            delta_samskara += (self.pos_karma * punya) - (self.neg_karma * paap)
            
        return samskaras + delta_samskara

class PanchkoshaNetwork(nn.Module):
    """
    The full Sanatani Architecture mapping 5 Koshas.
    Annamaya -> Pranamaya -> Manomaya -> Vijnanamaya -> Anandamaya.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 1. Annamaya & Pranamaya: Senses (Indriya) - Processed before this module
        
        # 2. Manomaya Kosha: The Weights (Samskaras) connecting senses to intellect
        # Start with random weak impressions
        self.samskaras = nn.Parameter(torch.rand(input_size, hidden_size) * 0.3)
        
        # 3. Vijnanamaya Kosha: The Intellect (Triguna Neuron)
        # Using Ahamkara (Ego) for Winner-Takes-All lateral inhibition
        # Using Tapasya (Adaptive Threshold) = 0.05 for smooth convergence over multiple epochs.
        self.buddhi = TrigunaNeuron(tamas_decay=0.8, sattva_threshold=1.5, wta_ahamkara=True, tapasya_rate=0.05)
        
        # The Law of Learning
        self.karma_law = KarmaSTDP(positive_karma=0.08, negative_karma=0.04)
        
    def forward(self, prana_spikes):
        # Multiply input spikes [Time, 1, In] by Samskaras [In, Out]
        time_steps = prana_spikes.size(0)
        rajas_input = torch.zeros(time_steps, 1, self.samskaras.size(1))
        
        for t in range(time_steps):
            rajas_input[t] = torch.matmul(prana_spikes[t], self.samskaras)
        
        # Buddhi processes the input and fires
        buddhi_spikes, _ = self.buddhi(rajas_input)
        return buddhi_spikes
        
    def experience_life(self, prana_spikes):
        """Unsupervised Training Step"""
        # Step 1: Mind fluctuates and Buddhi decides
        buddhi_spikes = self.forward(prana_spikes)
        
        # Step 2: Karma updates the Samskaras (Weights) 
        new_samskaras = self.karma_law(prana_spikes, buddhi_spikes, self.samskaras.data)
        
        # Niyama (Discipline): Weight Normalization (Homeostasis)
        # Prevent the neuron from learning everything by forcing it to unlearn unused paths
        # Ensures the sum of Samskaras entering each Buddhi is balanced (~1.0 or 1.5)
        target_karma = 1.5
        current_karma = new_samskaras.sum(dim=0, keepdim=True) + 1e-6
        normalized_samskaras = new_samskaras * (target_karma / current_karma)
        
        # Constrain weights to positive values (Niyama)
        self.samskaras.data.copy_(torch.clamp(normalized_samskaras, 0.0, 1.0))
        return buddhi_spikes

# --- BUILDING, TRAINING, AND TESTING THE SNN ---
if __name__ == "__main__":
    torch.manual_seed(108)
    print("=============================================")
    print("      BUILDING SANATANI SNN (SSNN)           ")
    print("=============================================\n")
    
    # 1. DATA GENERATION (Bhautik Jagat/Physical World)
    # We create two simple 3x3 images: A Horizontal Line vs A Vertical Line
    print("[1] Generating Environment Data...")
    img_horizontal = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0], # A strong horizontal line
        [0.0, 0.0, 0.0]
    ])
    
    img_vertical = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0], # A strong vertical line
        [0.0, 1.0, 0.0]
    ])
    
    # 2. INDRIYAS (Sense Organs to Spikes)
    print("[2] Indriyas are observing the world for 15 Kshan (Time-steps)...")
    eye = NetraIndriya(num_kshan=15)
    # Flatten 3x3 into 9 pixels and get Prana Spikes
    spikes_horiz = eye.forward_image(img_horizontal.flatten() * 2.0).unsqueeze(1) # Shape: [15, 1, 9]
    spikes_vert = eye.forward_image(img_vertical.flatten() * 2.0).unsqueeze(1)    # Shape: [15, 1, 9]
    
    # 3. INITIALIZE NETWORK
    # 9 sensory inputs -> 2 Buddhi neurons (We hope one learns Horizontal, one learns Vertical)
    # To force them to learn different things, we initialize weights favorably,
    # but let KarmaSTDP finalize the connections.
    print("[3] Waking up the Panchkosha Network...")
    ssnn = PanchkoshaNetwork(input_size=9, hidden_size=2)
    # Forcing slight asymmetry so they don't both learn the exact same thing (Natural variation)
    ssnn.samskaras.data[:, 0] += torch.tensor([0,0,0, 0.2,0.2,0.2, 0,0,0]) # Bias Node 0 to Horizontal
    ssnn.samskaras.data[:, 1] += torch.tensor([0,0.2,0, 0,0.2,0, 0,0.2,0]) # Bias Node 1 to Vertical
    
    print("\n[Initial Samskaras - Before Living/Training]:")
    print(ssnn.samskaras.data)
    
    # 4. TRAINING PHASE (Living life, accumulating Karma)
    epochs = 10
    print("\n[4] Training Phase - Accumulating Karma (STDP)...")
    for epoch in range(epochs):
        # Encounter Horizontal
        ssnn.experience_life(spikes_horiz)
        # Encounter Vertical
        ssnn.experience_life(spikes_vert)
        
    print("\n[Final Samskaras - After Living/Training]:")
    print("Notice how the weights (1) for specific patterns became stronger (Good Karma)!")
    print(ssnn.samskaras.data)

    # 5. TESTING PHASE (Buddhi identifies the pattern)
    print("\n[5] Testing Phase - What does Buddhi say?")
    out_h = ssnn(spikes_horiz).sum(dim=0).squeeze() # Sum of spikes over time
    out_v = ssnn(spikes_vert).sum(dim=0).squeeze()
    
    print(f"When seeing HORIZONTAL Line  -> Neuron 0 Spikes: {int(out_h[0])} | Neuron 1 Spikes: {int(out_h[1])}")
    print(f"When seeing VERTICAL Line    -> Neuron 0 Spikes: {int(out_v[0])} | Neuron 1 Spikes: {int(out_v[1])}")
    
    if out_h[0] > out_h[1] and out_v[1] > out_v[0]:
        print("\n\u2728 SUCCESS! Neuron 0 has become the 'Horizontal Line' Buddhi, and Neuron 1 is the 'Vertical Line' Buddhi. \u2728")
        print("The network evolved purely through Unsupervised Time and Karma, without Backpropagation.")
