from .utils import shift_sequence

def apply_audio_shift(audio, shift):
    return shift_sequence(audio, shift)

def apply_video_shift(video, shift):
    return shift_sequence(video, shift)

