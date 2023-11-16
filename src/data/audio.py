import librosa
import librosa.display
import warnings

def extract_tempo_and_beats(video_url_path):
     with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_signal, sample_rate = librosa.load(video_url_path)
        tempo, beat_frames = librosa.beat.beat_track(y=y_signal, sr=sample_rate)
        return tempo, beat_frames

def extract_rms_energy(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y)
        return rms

def extract_zero_crossing_rate(audio_path):
    y, sr = librosa.load(audio_path)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    return zcr