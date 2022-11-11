import os
from io import BytesIO

from scipy.io.wavfile import write
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
import speech_recognition as sr

os.environ["PATH"] += ":/Users/tomaszpionka/audio-orchestrator-ffmpeg/bin"


# pyglet.options['audio'] = ('openal', 'pulse', 'xaudio2', 'directsound', 'silent')


def wav_to_str(filepath: str, filepath_output: str, chunksize=60000):
    sound = AudioSegment.from_wav(filepath)

    def divide_chunks(sound, chunksize):
        # looping till length l
        for i in range(0, len(sound), chunksize):
            yield sound[i:i + chunksize]

    chunks = list(divide_chunks(sound, chunksize))
    r = sr.Recognizer()
    string_index = {}
    speech_string = ""
    for index, chunk in enumerate(chunks):
        chunk.export(f'{filepath_output}', format='wav')
        with sr.AudioFile(f'{filepath_output}') as source:
            audio = r.record(source)
            try:
                speech_string = r.recognize_google(audio, language='pl-PL')
            except:
                str_to_speech("Błąd! Nie zarejestrowano dźwięku.")
        print(speech_string)
        string_index[index] = speech_string
        break
    return speech_string


def speech_to_wav(filepath: str):
    fs = 16000  # Sample rate
    seconds = 8  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print("======= recording start =======")
    sd.wait()  # Wait until recording is finished
    write(f'{filepath}', fs, myrecording)  # Save as WAV file
    print("======= audio stop =======")


def speech_to_str(filepath: str, filepath_output: str):
    speech_to_wav(filepath)
    speech_str = wav_to_str(filepath, filepath_output)
    return speech_str


def str_to_speech(speech: str):
    tts = gTTS(text=speech, lang='pl', slow=False)
    speech_stream = BytesIO()
    tts.write_to_fp(speech_stream)
    speech_stream.seek(0)
    tts = AudioSegment.from_file(speech_stream, format="mp3")
    print("======= audio start =======")
    play(tts)
    print("======= audio stop =======")


if __name__ == "__main__":
    pass
