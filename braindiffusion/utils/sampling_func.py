import numpy as np

def convert_waveform(waveform, n, max_sampledepth=255):
    """
    Convert a waveform with positive and negative values ​​from -n to +n to a waveform with values ​​in the range 0-max_sampledepth.
    
    parameter:
    waveform: A waveform of shape (bs, 22, 750).
    n: The maximum amplitude of the wave, that is, the amplitude range of the wave is [-n, n].
    max_sampledepth: The maximum sample depth of the converted waveform, the default is 255.
    
    return:
    A waveform of shape (bs, 22, 750) with amplitudes in the range [0, max_sampledepth].
    """
    # 将波形值从 [-n, n] 映射到 [0, 1]
    waveform_normalized = (waveform + n) / (2 * n)
    
    # 将波形值从 [0, 1] 映射到 [0, max_sampledepth]
    waveform_scaled = waveform_normalized * max_sampledepth

    # waveform_scaled = waveform_scaled.astype(np.int16)
    
    return waveform_scaled

def invert_waveform(waveform_scaled, n, max_sampledepth=255):
    """
    Inverts the waveform from the 0-max_sampledepth value range back to a waveform with positive and negative values ​​from -n to +n.
    
    parameter:
    waveform_scaled: A waveform of shape (bs, 22, 750) with amplitudes in the range [0, max_sampledepth].
    n: The maximum amplitude of the wave, that is, the amplitude range of the wave is [-n, n].
    max_sampledepth: The maximum sample depth of the waveform before conversion, the default is 255.
    
    return:
    A waveform of shape (bs, 22, 750) with amplitudes in the range [-n, n].
    """
    waveform_normalized = waveform_scaled.astype(np.float32) / max_sampledepth
    
    # 将波形值从 [0, 1] 映射到 [-n, n]
    waveform = waveform_normalized * (4 * n)
    
    return waveform