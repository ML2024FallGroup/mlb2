import subprocess
import librosa
import math

def cut_audio(input_file, output_file, start = 0, duration=30):
    """Cuts the'duration' seconds of an audio file using ffmpeg."""

    cmd = [
        "ffmpeg",
        '-ss',str(start),
        "-i", input_file,
        "-t", str(duration),
        "-c", "copy",
        output_file,
        '-loglevel', 'panic',
          '-y'
    ]

    subprocess.run(cmd)


def chop_up(audio_file, duration=30, care=True, directory = 'data/audio/chop'):
  """chop audio file into 'duration' segments

  Args:
      audio_file (_type_): audio file to be choped
      duration (int, optional): chop duration. Defaults to 30.
      care (bool, optional): should we care about the extra endings. Defaults to True.
      directory (str, optional): directory to store chopped up files. Defaults to 'data/audio/chop'.
  """
  length = math.floor(librosa.get_duration(path=audio_file))
  print(length)
  filename, extension = audio_file.split('/')[-1].split('.')
  perfect = not length % duration
  counter = 1
  for start in range(0,length, duration):
    if not start + duration > length:
      cut_audio(audio_file, f'{directory}/{filename}_{counter}.{extension}',start)
    elif care:
     print(length-duration)
     cut_audio(audio_file, f'{directory}/{filename}_{counter}.{extension}',length-duration)
    counter += 1
     


  
    

