import wave

import pyaudio


class RecordAudio:
    def __init__(self):
        # 录音参数
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000

        # 打开录音
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    def record(self, output_path="audio/temp.wav", record_seconds=3):
        """
        recording
        :param output_path: The path where the recording is saved，siffix iswav
        :param record_seconds: recording time，Default 3 seconds
        :return: Recording file path
        """
        i = input("press enter to start recording，During 3 seconds of recording：")
        print("Start Recording...")
        frames = []
        for i in range(0, int(self.rate / self.chunk * record_seconds)):
            data = self.stream.read(self.chunk)
            frames.append(data)

        print("The recording has ended")
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        return output_path
