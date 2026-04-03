import torch
import torch.nn as nn

class TrigunaNeuron(nn.Module):
    """
    Sanatani Spiking Neuron Model.
    Models 'Man' (fluctuating mind), 'Buddhi' (intellect/decision), 
    processed over 'Kshan' (discrete moments).
    """
    def __init__(self, tamas_decay=0.9, sattva_threshold=1.0, wta_ahamkara=False, tapasya_rate=0.0):
        super(TrigunaNeuron, self).__init__()
        
        # Tamas represents Inertia / Resistance (Leak in LIF model)
        # Keeps the mind stable by naturally decaying the membrane potential
        self.tamas_decay = tamas_decay 
        
        # Sattva represents Clarity / Balance (Threshold in LIF model)
        # When 'Man' reaches Sattva, 'Buddhi' takes a clear decision (Spike)
        self.sattva_threshold = sattva_threshold
        
        # Ahamkara (Ego) - Winner Takes All Lateral Inhibition
        # The 'I' maker. If one Buddhi decides, it suppresses the others!
        self.wta_ahamkara = wta_ahamkara

        # Tapasya (Asceticism/Discipline) - Adaptive Threshold
        # The more a Buddhi fires, the higher its threshold becomes.
        # It forces the greedy neuron to let other silent neurons fire.
        self.tapasya_rate = tapasya_rate
        self.dynamic_threshold = None
        
    def forward(self, rajas_input_sequence):
        """
        Processes sensory inputs (Rajas/Activity) over time (Kshan).
        
        Args:
            rajas_input_sequence: Input current coming into the neuron over time. 
                                  Shape: [num_kshan (time_steps), batch_size, features]
        Returns:
            buddhi_spikes: The decisions made by Buddhi over time.
            man_history: History of Man's state (membrane potential).
        """
        num_kshan = rajas_input_sequence.size(0)
        batch_size = rajas_input_sequence.size(1)
        features = rajas_input_sequence.size(2)
        
        # Man (Mind/Membrane Potential) starts empty/calm (Shunya state)
        # It fluctuates as it experiences the world.
        man_potential = torch.zeros(batch_size, features)
        
        buddhi_spikes = []
        man_history = []
        
        # Initialize dynamic threshold if using Tapasya
        if self.tapasya_rate > 0.0:
            if self.dynamic_threshold is None or self.dynamic_threshold.size() != (batch_size, features):
                # Start threshold at base Sattva level
                self.dynamic_threshold = torch.full((batch_size, features), self.sattva_threshold, device=rajas_input_sequence.device)

        # Flow of Time (The cycle of Kshan)
        for kshan in range(num_kshan):
            current_rajas = rajas_input_sequence[kshan]
            
            # Use dynamic threshold if Tapasya is active, else base Sattva
            actual_threshold = self.dynamic_threshold if self.tapasya_rate > 0.0 else self.sattva_threshold

            # The fluctuation of Mind: 
            # 1. Experiences Decay due to Tamas (forgetting/inertia)
            # 2. Gets excited due to Rajas (stimulus/input)
            man_potential = (self.tamas_decay * man_potential) + current_rajas
            
            # Buddhi steps in when clarity (Sattva Threshold) is reached
            # If Man Potential >= Sattva -> 1 (Spike/Decision), else 0 (Indecision)
            if self.wta_ahamkara:
                # Ahamkara (Ego/Competition): Only the strongest Buddhi fires!
                spike = torch.zeros_like(man_potential)
                mask = man_potential >= actual_threshold
                if mask.any():
                    # Find who has the highest clarity (potential)
                    max_vals, max_indices = torch.max(man_potential, dim=1, keepdim=True)
                    # They only spike if they are the max AND crossed the threshold
                    spike.scatter_(1, max_indices, 1.0)
                    spike = spike * mask.float() # ensure it actually crossed threshold
                    
                    # The winner resets, and suppresses others slightly (Lateral Inhibition)
                    man_potential = man_potential - (spike * actual_threshold)
                    # Suppress the losers heavily (Ahamkara pushes others down)
                    man_potential = man_potential - (1.0 - spike) * 0.8
            else:
                # Standard independent spiking
                spike = (man_potential >= actual_threshold).float()
                # After a pure decision (Karma), the mind is mostly reset, clearing that exact thought
                # Soft reset mechanism: subtracting threshold value from potential
                man_potential = man_potential - (spike * actual_threshold)
                
            # If Tapasya is active, firing increases the threshold (harder to fire next time)
            # Small decay brings threshold slowly back to base over time if inactive
            if self.tapasya_rate > 0.0:
                self.dynamic_threshold += spike * self.tapasya_rate
                self.dynamic_threshold -= (self.dynamic_threshold - self.sattva_threshold) * 0.005
            
            buddhi_spikes.append(spike)
            man_history.append(man_potential.clone())
            
        # Returning the sequence of spikes (Karma generated) and history of mind
        return torch.stack(buddhi_spikes), torch.stack(man_history)

# Quick Sanity Test
if __name__ == "__main__":
    print("Initializing Sanatani Neural Network components...")
    
    # 5 Kshan (time steps), 1 sample in batch, 3 input features (senses)
    num_kshan = 10
    batch_size = 1
    senses = 3
    
    # Creating a dummy sequence of chaotic sensory input (Rajas)
    torch.manual_seed(108) # Auspicious seed
    sensory_inputs = torch.rand((num_kshan, batch_size, senses)) * 1.5
    
    # Instantiate the Neuron
    neuron = TrigunaNeuron(tamas_decay=0.8, sattva_threshold=1.0)
    
    # Run the simulation over all Kshan
    spikes, mind_states = neuron(sensory_inputs)
    
    print("\n--- RESULTS ---")
    print(f"Total Spikes (Decisions by Buddhi):\n {spikes.sum(dim=0).squeeze()}")
    print("\nMind State at final Kshan:\n", mind_states[-1].squeeze())
