import torch
import torch.nn as nn

class Gyanendriya(nn.Module):
    """
    Base class for Sensary Organs (Gyanendriyas).
    Converts physical matter (Data) into Prana (Spikes) over Kshan (Time).
    """
    def __init__(self, num_kshan=10):
        super(Gyanendriya, self).__init__()
        self.num_kshan = num_kshan # How many time steps (Kshan) to generate

class NetraIndriya(Gyanendriya):
    """
    The Eye (Netra) - Image/Video Encoder (Prakash & Gati)
    Uses Poisson Rate Coding for Static Images and Delta Coding for Video.
    """
    def forward_image(self, image_tensor):
        """
        Converts a static image (2D/3D) into a Spike Train (Time, C, H, W).
        Higher pixel intensity (Jyoti/Brightness) = Higher probability of a spike.
        """
        # Add a time dimension (Kshan)
        # We pretend we are "looking" at the image for `num_kshan` moments
        shape = (self.num_kshan, *image_tensor.shape)
        # Generate random numbers, if pixel > rand -> Spike (1.0)
        random_matrix = torch.rand(shape, device=image_tensor.device)
        # Normalize image to 0-1 if not already
        normalized_img = image_tensor / image_tensor.max()
        # The bright pixels will easily beat the random numbers, firing more often
        spikes = (normalized_img.unsqueeze(0) > random_matrix).float()
        return spikes

    def forward_video(self, video_frames):
        """
        Converts a video (Time, C, H, W) to Spikes.
        Only fires when something changes (Delta/Event Coding). This is how real eyes work!
        If a pixel stays the same, it stops firing (Tamas/Adaptation).
        """
        spikes = []
        # Frame at Kshan 0 is the baseline
        prev_frame = video_frames[0]
        spikes.append((prev_frame > 0.5).float()) # Initial burst
        
        # Look for changes across Kshan
        for kshan in range(1, video_frames.size(0)):
            curr_frame = video_frames[kshan]
            # If the change (Delta) is significant > threshold
            delta = curr_frame - prev_frame
            spike = (torch.abs(delta) > 0.1).float() 
            spikes.append(spike)
            prev_frame = curr_frame
        return torch.stack(spikes)

class KarnaIndriya(Gyanendriya):
    """
    The Ear (Karna) - Audio Encoder (Naad / Shabda)
    Takes a spectrogram (Frequency over Time) and converts to Spikes.
    """
    def forward(self, spectrogram):
        """
        A spectrogram already has a time dimension and frequency dimension.
        We convert the intensity of the sound at a given frequency into spikes.
        """
        # Normalize and use Poisson rate coding 
        normalized_audio = spectrogram / spectrogram.max()
        random_matrix = torch.rand_like(normalized_audio)
        # If the sound is loud at a frequency, it fires more spikes
        spikes = (normalized_audio > random_matrix).float()
        return spikes

class VaniIndriya(Gyanendriya):
    """
    Speech / Text (Vani / Lipi) - Temporal Sequence Encoder
    Words play out over time (Kshan).
    """
    def forward(self, word_embeddings):
        """
        Takes sequential embeddings and converts them to spikes via thresholding.
        word_embeddings shape: [sequence_length, batch, embedding_dim]
        """
        # A simple thresholding mechanism for semantic spikes
        # Only strong features in the embedding form a thought/spike
        spikes = (word_embeddings > 0.5).float()
        return spikes

# Testing the Indriyas (Sense Organs)
if __name__ == "__main__":
    import math
    print("Testing Gyanendriyas (Sense Organs)...")
    
    # 1. 👁️ Netra (Eye) - Image 
    eye = NetraIndriya(num_kshan=5)
    dummy_image = torch.tensor([[0.1, 0.9], [0.8, 0.0]]) # 2x2 image
    image_spikes = eye.forward_image(dummy_image)
    
    # The pixel with 0.9 (bright) will fire almost every Kshan. The 0.1 pixel will rarely fire.
    print(f"\nImage Spikes over 5 Kshan (Notice the bright pixel [0,1] fires more):\n{image_spikes.squeeze()}")
    
    # 2. 🦻 Karna (Ear) - Audio 
    ear = KarnaIndriya()
    # Dummy Spectrogram: [Time/Kshan=3, Frequencies=4]
    dummy_spectrogram = torch.rand((3, 4))
    audio_spikes = ear.forward(dummy_spectrogram)
    print(f"\nAudio Spikes for 3 Kshan:\n{audio_spikes}")
