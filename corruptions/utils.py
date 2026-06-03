import numpy as np

def shift_sequence(seq, shift):
    """Shift a sequence by a given amount.

    Args:
        seq (np.ndarray): The input sequence to be shifted.
        shift (int): The number of positions to shift the sequence. Positive values shift to the right, negative values shift to the left.

    Returns:
        np.ndarray: The shifted sequence.
    """

    T, D = seq.shape

    shifted =  np.zeros_like(seq)

    if shift > 0:
        shifted[shift:] = seq[:-shift]

    elif shift < 0:
        shifted[:shift] = seq[-shift:]
    
    else:
        shifted = seq.copy()

    return shifted


x = np.arange(20).reshape(10, 2)

print("Original sequence:")
print(x)
print("\nShifted sequence (shift=2):")
print(shift_sequence(x, 2))