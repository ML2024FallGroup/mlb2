import subprocess

def cut_audio(input_file, output_file, duration=30):
    """Cuts the first 'duration' seconds of an audio file using ffmpeg."""

    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-t", str(duration),
        "-c", "copy",
        output_file
    ]

    subprocess.run(cmd)

# cut_audio("aud1.mp3", "output.mp3", 30)


def separate_stems(audio_file, output_dir, model_type="spleeter:2stems"):
    """
    Separates stems from an audio file using Spleeter.

    Args:
        audio_file (str): Path to the input audio file.
        output_dir (str): Path to the directory where the separated stems will be saved.
        model_type (str, optional): Spleeter model to use. Defaults to 'spleeter:2stems'.
                                    Available options: 'spleeter:2stems', 'spleeter:4stems', 'spleeter:5stems'.        """

    cmd = [
        "spleeter",
        'separate',
         audio_file,
        "-p", model_type,
        "-o", output_dir
    ]

    subprocess.call(cmd)

if __name__ == "__main__":
    audio_file = "out.mp3"  # Replace with the path to your audio file
    output_dir = "output_folder"  # Replace with the desired output directory
    separate_stems(audio_file, output_dir, "spleeter:4stems")