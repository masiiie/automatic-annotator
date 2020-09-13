import os
import numpy as np
from scipy.io import wavfile

from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Normalize,
    AddImpulseResponse,
    FrequencyMask,
    TimeMask,
    AddGaussianSNR,
    Resample,
    ClippingDistortion,
    AddBackgroundNoise,
    AddShortNoises)

DEMO_DIR = './augmentation utils'
#SAMPLE_RATE = 16000
SAMPLE_RATE = 44100
CHANNELS = 1

def load_wav_file(sound_file_path):
    sample_rate, sound_np = wavfile.read(sound_file_path)
    if sample_rate != SAMPLE_RATE:
        raise Exception(
            "Unexpected sample rate {} (expected {})".format(sample_rate, SAMPLE_RATE)
        )

    if sound_np.dtype != np.float32:
        assert sound_np.dtype == np.int16
        sound_np = sound_np / 32767  # ends up roughly between -1 and 1

    return sound_np
    


def transform(file_path, output_folder, iterations):
    """
    For each transformation, apply it to an example sound and write the transformed sounds to
    an output folder.
    """

    samples = load_wav_file(file_path)
    file_name = os.path.basename(file_path).replace('.wav','')

    def produce(augmenter, name):
        for i in range(iterations):
            output_file_path = '{}/{}'.format(output_folder, "{}_{}_{}.wav".format(name, file_name, i))
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
            wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)


    # TimeMask
    augmenter = Compose([TimeMask(p=1.0)])
    produce(augmenter,'TimeMask')


    # FrequencyMask
    augmenter = Compose([FrequencyMask(p=1.0)])
    produce(augmenter , 'FrequencyMask')


    # AddGaussianSNR
    augmenter = Compose([AddGaussianSNR(p=1.0)])
    produce(augmenter, 'AddGaussianSNR')

    # PitchShift
    augmenter = Compose([PitchShift(min_semitones=-4, max_semitones=4, p=1.0)])
    produce(augmenter, 'PitchShift')

    # TimeStretch
    augmenter = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)])
    produce(augmenter, 'TimeStretch')

    # AddGaussianNoise
    augmenter = Compose(
        [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)]
    )
    produce(augmenter, 'AddGaussianNoise')

    # Shift
    augmenter = Compose([Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0)])
    produce(augmenter, 'Shift')



    # Shift without rollover
    augmenter = Compose(
    [Shift(min_fraction=-0.5, max_fraction=0.5, rollover=False, p=1.0)])
    produce(augmenter, 'Shift without rollover')


    # Normalize
    augmenter = Compose([Normalize(p=1.0)])
    produce(augmenter, 'Normalize')


    # AddImpulseResponse
    augmenter = Compose(
        [AddImpulseResponse(p=1.0, ir_path=os.path.join(DEMO_DIR, "ir"))]
    )
    produce(augmenter, 'AddImpulseResponse')


    # Resample
    augmenter = Compose([Resample(p=1.0)])
    produce(augmenter, 'Resample')


    # ClippingDistortion
    augmenter = Compose([ClippingDistortion(p=1.0)])
    produce(augmenter, 'ClippingDistortion')


    # AddBackgroundNoise
    augmenter = Compose(
        [
            AddBackgroundNoise(
                sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0
            )
        ]
    )
    produce(augmenter, 'AddBackgroundNoise')


    # AddShortNoises
    augmenter = Compose(
        [
            AddShortNoises(
                sounds_path=os.path.join(DEMO_DIR, "short_noises"),
                min_snr_in_db=0,
                max_snr_in_db=8,
                min_time_between_sounds=2.0,
                max_time_between_sounds=4.0,
                burst_probability=0.4,
                min_pause_factor_during_burst=0.01,
                max_pause_factor_during_burst=0.95,
                min_fade_in_time=0.005,
                max_fade_in_time=0.08,
                min_fade_out_time=0.01,
                max_fade_out_time=0.1,
                p=1.0,
            )
        ]
    )
    produce(augmenter, 'AddShortNoises')