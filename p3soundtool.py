from pydub import AudioSegment


# 將mp3轉成Wav及輸出成smaple rate:16000
def mp3_to_wav(mp3filepath, wavfilepath):
    import wave
    import io

    sound = AudioSegment.from_file(mp3filepath, format='mp3')
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    sound.export(wavfilepath, format="wav")


def cut_wav_from(wavobj, savewavpath, start_time, end_time):
    #wordSeg = wavobj[float(start_time) * 1000: (float(end_time)) * 1000]
    wordSeg = wavobj[float(start_time): (float(end_time))]
    wordSeg = wordSeg.set_frame_rate(16000)
    wordSeg = wordSeg.set_channels(1)
    wordSeg.export(savewavpath, format='wav')
