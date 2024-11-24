from os import path
import subprocess

from django.conf import settings
import librosa
import math
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path

class AudioProcessor:
    """A class for processing and analyzing audio files."""
    
    DEFAULT_DURATION_SECONDS = 30
    DEFAULT_MFCC_PARAMS = {
        "n_mfcc": 13,
        "n_fft": 2048,
        "hop_length": 512,
        'sample_rate': 22050,
        'num_segments': 10,
        'samples_per_label': 20
    }

    @staticmethod
    def trim_audio(input_path: str, output_path: str, start_time: float = 0, 
                   duration: float = DEFAULT_DURATION_SECONDS) -> None:
        """
        Trims an audio file to specified duration using ffmpeg.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save trimmed audio
            start_time: Start time in seconds
            duration: Duration to trim in seconds
        """
        ffmpeg_command = [
            "ffmpeg",
            '-ss', str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c", "copy",
            output_path,
            '-loglevel', 'panic',
            '-y'
        ]
        subprocess.run(ffmpeg_command)

    @staticmethod
    def separate_stems(input_path: str, output_dir: str, model_type: str = "spleeter:4stems") -> None:
        """
        Separates an audio file into stems using Spleeter.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory where separated stems will be saved
            model_type: Spleeter model configuration to use
                Available options:
                - 'spleeter:2stems' (vocals + accompaniment)
                - 'spleeter:4stems' (vocals + drums + bass + other)
                - 'spleeter:5stems' (vocals + drums + bass + piano + other)
                
        Raises:
            ValueError: If model_type is not one of the supported options
        """
        valid_models = {"spleeter:2stems", "spleeter:4stems", "spleeter:5stems"}
        if model_type not in valid_models:
            raise ValueError(f"Model type must be one of {valid_models}")
            
        spleeter_command = [
            "spleeter",
            'separate',
            input_path,
            "-p", model_type,
            "-o", output_dir
        ]
        
        subprocess.run(spleeter_command, check=True)

    @staticmethod
    def segment_audio(audio_path: str, segment_duration: int = DEFAULT_DURATION_SECONDS, 
                     keep_remainder: bool = True) -> None:
        """
        Segments an audio file into fixed-duration chunks.
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            keep_remainder: Whether to keep the remainder segment
        """
        output_dir = Path(settings.MEDIA_ROOT)/ 'audio' / 'segments' / audio_path.split('/')[-1]
        output_dir.mkdir(parents=True, exist_ok=True)
        total_duration = math.floor(librosa.get_duration(path=audio_path))
        filename, extension = audio_path.split('/')[-1].split('.')
        segment_count = 1
        
        for start_time in range(0, total_duration, segment_duration):
            output_path = f'{output_dir}/{filename}_{segment_count}.{extension}'
            
            if start_time + segment_duration <= total_duration:
                AudioProcessor.trim_audio(audio_path, output_path, start_time)
            elif keep_remainder:
                start_time = total_duration - segment_duration
                AudioProcessor.trim_audio(audio_path, output_path, start_time)
                break
            segment_count += 1

    @staticmethod
    def extract_mfcc_features(audio_path: str, params: Dict[str, Any] = None) -> np.ndarray:
        """
        Extracts MFCC features from an audio file.
        
        Args:
            audio_path: Path to audio file
            params: MFCC extraction parameters
            
        Returns:
            numpy.ndarray: MFCC features matrix
        """
        if params is None:
            params = AudioProcessor.DEFAULT_MFCC_PARAMS
            
        audio_data, _ = librosa.load(audio_path)
        samples_per_track = AudioProcessor.DEFAULT_DURATION_SECONDS * params['sample_rate']
        samples_per_segment = samples_per_track // params['num_segments']

        mfcc_features = librosa.feature.mfcc(
            y=audio_data,
            sr=params['sample_rate'],
            n_mfcc=params["n_mfcc"],
            n_fft=params["n_fft"],
            hop_length=params["hop_length"]
        )

        expected_vectors = math.ceil(samples_per_segment / params["hop_length"])
        if mfcc_features.shape[1] != expected_vectors:
            mfcc_features = np.resize(mfcc_features, (params["n_mfcc"], expected_vectors))
        
        return mfcc_features

    @staticmethod
    def visualize_mfcc(audio_path: str, output_path: str) -> None:
        """
        Generates and saves MFCC visualization.
        
        Args:
            audio_path: Path to audio file
            output_path: Path to save visualization
        """
        plt.figure(figsize=(10, 4))
        mfcc_features = AudioProcessor.extract_mfcc_features(audio_path)
        librosa.display.specshow(mfcc_features, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC Visualization')
        plt.tight_layout()
        plt.set_cmap('viridis')
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def extract_audio_features(audio_path: str) -> Dict[str, float]:
        """
        Extracts comprehensive audio features including spectral and temporal characteristics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict[str, float]: Dictionary of extracted features
        """
        audio_data, sample_rate = librosa.load(audio_path)
        filename = audio_path.split('/')[-1]
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        
        features = {
            'filename': filename,
            'length': duration
        }
        
        # Spectral Features
        features.update(AudioProcessor._extract_spectral_features(audio_data, sample_rate))
        
        # Temporal Features
        features.update(AudioProcessor._extract_temporal_features(audio_data, sample_rate))
        
        # MFCC Features
        features.update(AudioProcessor._extract_mfcc_statistics(audio_data, sample_rate))
        
        return features

    @staticmethod
    def _extract_spectral_features(audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Helper method to extract spectral features."""
        features = {}
        
        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)
        
        # Spectral Features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_var'] = np.var(spectral_centroids)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)
        
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_var'] = np.var(rolloff)
        
        return features

    @staticmethod
    def _extract_temporal_features(audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Helper method to extract temporal features."""
        features = {}
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio_data)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        
        # Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
        features['zero_crossing_rate_var'] = np.var(zero_crossing_rate)
        
        # Harmony and Percussive Components
        harmonic, percussive = librosa.effects.hpss(audio_data)
        features['harmony_mean'] = np.mean(harmonic)
        features['harmony_var'] = np.var(harmonic)
        features['perceptr_mean'] = np.mean(percussive)
        features['perceptr_var'] = np.var(percussive)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        features['tempo'] = tempo
        
        return features

    @staticmethod
    def _extract_mfcc_statistics(audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Helper method to extract MFCC statistics."""
        features = {}
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
        
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_var'] = np.var(mfccs[i])
            
        return features

    @staticmethod
    def process_audio_to_dataframe(audio_path: str) -> pd.DataFrame:
        """
        Processes audio file and returns features as a DataFrame.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            pd.DataFrame: DataFrame containing extracted features
        """
        features = AudioProcessor.extract_audio_features(audio_path)
        df = pd.DataFrame([features])
        
        # Define column order
        column_order = [
            'filename', 'length', 
            'chroma_stft_mean', 'chroma_stft_var',
            'rms_mean', 'rms_var',
            'spectral_centroid_mean', 'spectral_centroid_var',
            'spectral_bandwidth_mean', 'spectral_bandwidth_var',
            'rolloff_mean', 'rolloff_var',
            'zero_crossing_rate_mean', 'zero_crossing_rate_var',
            'harmony_mean', 'harmony_var',
            'perceptr_mean', 'perceptr_var',
            'tempo'
        ]
        
        # Add MFCC columns
        column_order.extend([f'mfcc{i+1}_mean' for i in range(20)])
        column_order.extend([f'mfcc{i+1}_var' for i in range(20)])
        
        return df[column_order]