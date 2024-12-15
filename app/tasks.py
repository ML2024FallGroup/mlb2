import os
from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import time
from pathlib import Path
from mlb2.app.audio_processing_ import AudioProcessor
import tensorflow as tf
import joblib
import numpy as np
import librosa

@shared_task
def run():
  channel_layer = get_channel_layer()
  for _ in range(11):
    async_to_sync(channel_layer.group_send)(
            'test',
            {
                'type': 'chat_message',
                'message': 'proceed'
            }
        )
    time.sleep(3)

label_dict = {
        0: 'blues',
        1: 'classical',
        2: 'country',
        3: 'disco',
        4: 'hiphop',
        5: 'jazz',
        6: 'metal',
        7: 'pop',
        8: 'reggae',
        9: 'rock'
    }
genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


@shared_task
def process_audio(audio_path):
    """Main task to process the uploaded audio file."""
    channel_layer = get_channel_layer()
    
    # Process main audio file
    process_main_audio.delay(audio_path)
    
    # Segment and process chunks
    segment_and_process.delay(audio_path)

@shared_task
def process_main_audio(audio_path):
    """Process and send data for the main audio file."""
    channel_layer = get_channel_layer()
    
    # Get metadata
    audio_meta = AudioProcessor.get_metadata(audio_path)
    
    cover_art = AudioProcessor.extract_cover_art(audio_path)
    # Generate spectrogram
    visual = AudioProcessor.visualize_mfcc(audio_path)
    
    # Send metadata
    async_to_sync(channel_layer.group_send)(
        'test',
        {
            'type': 'main_message',
            'item_type': 'main_audio_data',
            'detail': {
                'metadata': audio_meta,
                'spectrogram': str(visual),
                'audio_path': audio_path,
                'cover': cover_art
            }
        }
    )

@shared_task
def segment_and_process(audio_path):
    """Segment audio and process each segment."""
    channel_layer = get_channel_layer()
    # Segment the audio
    segment_paths = AudioProcessor.segment_audio(audio_path)
    result = []
    stems = ['vocals.wav', 'drums.wav', 'bass.wav', 'other.wav']
    for segment in segment_paths:
        directory = AudioProcessor.separate_stems(segment)
        visual = AudioProcessor.visualize_mfcc(segment)
        statiscal_prediction = make_statical_prediction(segment)
        cnn_prediction = make_genre_prediction(load_model(), segment, directory)
        stem_vis = []
        for stem in stems:
            stem_vis.append(AudioProcessor.visualize_mfcc(directory + '/' + stem,True))
        
        async_to_sync(channel_layer.group_send)(
            'test',
            {
                'type': 'stem_message',
                'item_type': 'stem_data',
                'detail': {
                    'segment_data': {"stem_director": directory, 'segment_spec': visual, 'stem_specs': stem_vis, 'segment_path': segment, 'dnn_pred': statiscal_prediction , 'cnn_prediction': cnn_prediction},
                }
            }
        )
        # process_stem.delay(stems, segment)

@shared_task
def process_stem(stems, segment):
    channel_layer = get_channel_layer()
    directory = AudioProcessor.separate_stems(segment)
    visual = AudioProcessor.visualize_mfcc(segment)
    statiscal_prediction = make_statical_prediction(segment)
    cnn_prediction = make_genre_prediction(load_model(), segment, directory)
    stem_vis = []
    for stem in stems:
        stem_vis.append(AudioProcessor.visualize_mfcc(directory + '/' + stem,True))
        
        async_to_sync(channel_layer.group_send)(
            'test',
            {
                'type': 'stem_message',
                'item_type': 'stem_data',
                'detail': {
                    'segment_data': {"stem_director": directory, 'segment_spec': visual, 'stem_specs': stem_vis, 'segment_path': segment, 'dnn_pred': statiscal_prediction , 'cnn_prediction': cnn_prediction},
                }
            }
        )

def predict_genre(model, scaler, features, label_dict):
    """
    Predict the genre of an audio file
    
    Parameters:
    model: Loaded Keras model
    scaler: Loaded StandardScaler
    audio_path: Path to the audio file
    label_dict: Dictionary mapping numerical predictions to genre labels
    
    Returns:
    predicted_genre: String name of the predicted genre
    probabilities: Dictionary of genre probabilities
    """
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    predictions = model.predict(scaled_features)
    predicted_class = np.argmax(predictions[0])
    predicted_genre = label_dict[predicted_class]
    
    # Get probabilities for each genre
    probabilities = {label_dict[i]: float(prob) for i, prob in enumerate(predictions[0])}
    
    return predicted_genre, probabilities

def make_statical_prediction(audio_file):

    features = AudioProcessor.process_audio_to_dataframe(audio_file)
    
    new_model = tf.keras.models.load_model('models/my_model.keras')
    scaler = joblib.load("models/my_scaler.pkl")


    _, probabilities = predict_genre(new_model, scaler, features, label_dict)
    return probabilities

def load_model(model_path="models/merged_genre_model.keras"):
    """
    Load the model from disk
    """
    return tf.keras.models.load_model(model_path)

def process_audio_file(file_path, duration=30, sample_rate=22050, num_segments=10):
    """Process a single audio file into MFCC segments."""
    # Constants from original training
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512

    # Calculate samples
    samples_per_track = duration * sample_rate
    samples_per_segment = samples_per_track // num_segments

    # Load audio file
    signal, sr = librosa.load(file_path, sr=sample_rate)

    # Process each segment
    feature_segments = []
    for segment in range(num_segments):
        start_sample = samples_per_segment * segment
        end_sample = start_sample + samples_per_segment
        signal_segment = signal[start_sample:end_sample]

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=signal_segment,
                                  sr=sample_rate,
                                  n_mfcc=n_mfcc,
                                  n_fft=n_fft,
                                  hop_length=hop_length)

        # Ensure consistent shape
        expected_vectors = np.ceil(samples_per_segment / hop_length)
        if mfcc.shape[1] != expected_vectors:
            mfcc = np.resize(mfcc, (n_mfcc, int(expected_vectors)))

        feature_segments.append(mfcc.T)

    return np.array(feature_segments)

def process_stems_directory(stems_dir, duration=30, sample_rate=22050, num_segments=10):
    """Process all stems in a directory."""
    # Get sorted list of stem files
    stem_files = sorted([f for f in os.listdir(stems_dir) if f.endswith('.wav') or f.endswith('.mp3')])

    # Process each stem file
    all_stem_features = []
    for stem_file in stem_files:
        stem_path = os.path.join(stems_dir, stem_file)
        stem_features = process_audio_file(stem_path, duration, sample_rate, num_segments)
        all_stem_features.append(stem_features)

    # Stack all stem features
    return np.stack(all_stem_features, axis=1)

def prepare_for_prediction(song_path, stems_dir):
    """Prepare audio data for prediction by processing both song and stems."""
    # Process original song
    song_features = process_audio_file(song_path)
    song_features = np.expand_dims(song_features, axis=1)  # Add stem dimension

    # Process stems
    stem_features = process_stems_directory(stems_dir)

    # Concatenate song and stem features
    combined_features = np.concatenate([song_features, stem_features], axis=1)

    # Move channel (stem) dimension to last axis for model input
    combined_features = np.moveaxis(combined_features, 1, -1)

    return combined_features

def make_genre_prediction(model, audio_path, stems_dir=None):
    """
    Predict the genre of an audio file using both original audio and stems if available

    Args:
        model: Loaded tensorflow model
        audio_path: Path to the original audio file
        stems_dir: Directory containing the stems (optional)

    Returns:
        Dictionary containing predictions and probabilities
    """
    features = prepare_for_prediction(audio_path, stems_dir)

    # Get predictions
    predictions = model.predict(features)

        # Average predictions across segments
    avg_predictions = np.mean(predictions, axis=0)

    # Sort predictions
    sorted_indices = np.argsort(avg_predictions)[::-1]

    # Create sorted list of (genre, probability) tuples
    results = {genre_labels[idx]: float(avg_predictions[idx])
              for idx in sorted_indices}

    return results