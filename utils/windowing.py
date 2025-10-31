import torch

def generate_synthesis_window(output_len: int, hop_len: int, analysis_window: torch.Tensor) -> torch.Tensor:
    """
    Generate a synthesis window for overlap-add reconstruction
    based on Wang et al., 2022.

    Args:
        output_len (int): Output window size in samples.
        hop_len (int): Hop size in samples.
        analysis_window (torch.Tensor): Input analysis window.

    Returns:
        torch.Tensor: Synthesis window of length `output_len`.
    """
    A = output_len
    B = hop_len
    N = len(analysis_window)

    synthesis_window = torch.zeros(A)
    for n in range(A):
        numerator = analysis_window[N - A + n]
        denominator = torch.zeros(1)
        for k in range(int(A / B)):
            denominator += analysis_window[N - A + (n % B) + k * B] ** 2
        synthesis_window[n] = numerator / denominator
    return synthesis_window
