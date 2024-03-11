import tkinter as tk
from tkinter import filedialog
import pyaudio
import numpy as np
import threading
from scipy.signal import resample, butter
import pickle
import soundfile as sf
from pydub import AudioSegment
import os

class VoiceChangerApp:
    def __init__(self, master):
        self.master = master
        master.title("JaleoBoard")

        # Label for effect selection
        self.label = tk.Label(master, text="Select an effect:")
        self.label.pack()

        # Dropdown menu for selecting effect
        self.effects = ["Robot", "Deep Voice", "Chipmunk"]  # Add more effects as needed
        self.effect_variable = tk.StringVar(master)
        self.effect_variable.set(self.effects[0])  # Default effect
        self.effect_menu = tk.OptionMenu(master, self.effect_variable, *self.effects, command=self.change_effect)
        self.effect_menu.pack()

        # Button for starting voice changer
        self.start_button = tk.Button(master, text="Start", command=self.start_voice_changer)
        self.start_button.pack()

        # Button for stopping voice changer
        self.stop_button = tk.Button(master, text="Stop", command=self.stop_voice_changer)
        self.stop_button.pack()
        self.stop_button.config(state=tk.DISABLED)  # Initially disabled

        # Volume slider
        self.volume_label = tk.Label(master, text="Volume:")
        self.volume_label.pack()
        self.volume_scale = tk.Scale(master, from_=0, to=100, orient=tk.HORIZONTAL, command=self.change_volume)
        self.volume_scale.set(80)  # Default volume level (adjust as needed)
        self.volume_scale.pack()

        # Add sound buttons
        self.custom_sounds = [None] * 10
        self.sound_frame1 = tk.Frame(master)
        self.sound_frame1.pack(side=tk.LEFT, padx=5, pady=5)
        self.sound_frame2 = tk.Frame(master)
        self.sound_frame2.pack(side=tk.LEFT, padx=5, pady=5)
        self.sound_frames = [self.sound_frame1, self.sound_frame2]

        for i in range(10):
            frame_idx = i // 5  # Determine which frame the button should go in
            frame = self.sound_frames[frame_idx]
            button_frame = tk.Frame(frame)
            button_frame.pack()
            play_button = tk.Button(button_frame, text=f"Sound {i + 1}", command=lambda idx=i: self.play_custom_sound(idx))
            play_button.grid(row=i % 5, column=0)
            config_button = tk.Button(button_frame, text="Config", command=lambda idx=i: self.select_custom_sound(idx))
            config_button.grid(row=i % 5, column=1)

        # Load custom sounds
        self.load_custom_sounds()

        # Audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.RATE = 44100

        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=1,
                                  rate=self.RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.CHUNK)

        # State variables
        self.active = False
        self.volume = 0.8  # Default volume level
        self.effect_function = self.robot_effect  # Set default effect
        self.audio_thread = None

        # Butterworth low-pass filter coefficients for smoothing
        self.b, self.a = butter(5, 0.05, btype='low')

        # Bind closing window event
        master.protocol("WM_DELETE_WINDOW", self.close_window)

    # Callback function for effect selection
    def change_effect(self, selected_effect):
        if selected_effect == "Robot":
            self.effect_function = self.robot_effect
        elif selected_effect == "Deep Voice":
            self.effect_function = self.deep_voice_effect
        elif selected_effect == "Chipmunk":
            self.effect_function = self.chipmunk_effect

    # Callback function for volume change
    def change_volume(self, volume):
        self.volume = float(volume) / 100.0

    # Start the voice changer
    def start_voice_changer(self):
        if not self.active:
            self.active = True
            self.start_button.config(state=tk.DISABLED)  # Disable start button
            self.stop_button.config(state=tk.NORMAL)  # Enable stop button
            self.audio_thread = threading.Thread(target=self.process_audio)  # Start audio processing in a separate thread
            self.audio_thread.start()

    # Stop the voice changer
    def stop_voice_changer(self):
        if self.active:
            self.active = False
            self.start_button.config(state=tk.NORMAL)  # Enable start button
            self.stop_button.config(state=tk.DISABLED)  # Disable stop button

    # Play custom sound
    def play_custom_sound(self, idx):
        if self.custom_sounds[idx] is not None:
            self.stream.write(self.custom_sounds[idx].tobytes())

    # Audio processing loop
    def process_audio(self):
        while self.active:
            data = self.stream.read(self.CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16)

            if self.effect_function:
                audio_data = self.effect_function(audio_data)

            # Apply volume adjustment
            audio_data = (audio_data.astype(np.float32) * self.volume).astype(np.int16)

            self.stream.write(audio_data.tobytes())

    # Effect function: Robot
    def robot_effect(self, audio_data):
        # Shift the pitch of the audio to simulate a robotic effect
        new_rate = int(self.RATE * 0.5)  # Decrease the sampling rate significantly for deeper effect
        audio_data_resampled = resample(audio_data, int(len(audio_data) * new_rate / self.RATE))

        # Apply a stronger pitch shift
        pitch_shift_factor = 0.3  # Adjust as needed
        audio_data_resampled = np.interp(np.arange(0, len(audio_data_resampled), pitch_shift_factor),
                                         np.arange(0, len(audio_data_resampled)), audio_data_resampled)

        # Apply aggressive thresholding to make the sound more binary-like
        threshold = np.median(audio_data_resampled)  # Calculate median value
        binary_audio = np.where(audio_data_resampled >= threshold * 0.5, threshold, -threshold)  # Apply threshold

        return binary_audio.astype(np.int16)

    # Effect function: Deep Voice
    def deep_voice_effect(self, audio_data):
        # Shift the pitch of the audio to simulate a deeper voice
        new_rate = int(self.RATE * 1.5)  # Increase the sampling rate slightly for higher pitch
        audio_data_resampled = resample(audio_data, int(len(audio_data) * new_rate / self.RATE))

        return audio_data_resampled.astype(np.int16)

    # Effect function: Chipmunk
    def chipmunk_effect(self, audio_data):
        # Shift the pitch of the audio to simulate a chipmunk-like voice
        new_rate = int(self.RATE * 0.7)  # Decrease the sampling rate slightly for deeper effect
        audio_data_resampled = resample(audio_data, int(len(audio_data) * new_rate / self.RATE))

        return audio_data_resampled.astype(np.int16)

    # Select custom sound file
    def select_custom_sound(self, idx):
        file_path = filedialog.askopenfilename(title="Select Sound File", filetypes=(("Audio files", "*.wav;*.mp3"), ("All files", "*.*")))
        if file_path:
            try:
                if file_path.endswith('.wav'):
                    data, rate = sf.read(file_path)
                else:  # Assume MP3
                    sound = AudioSegment.from_mp3(file_path)
                    data = np.array(sound.get_array_of_samples())
                    rate = sound.frame_rate
                if rate != self.RATE:
                    data = resample(data, int(len(data) * self.RATE / rate))
                self.custom_sounds[idx] = data.astype(np.int16)
            except Exception as e:
                print("Error loading custom sound:", e)

    # Load custom sounds from file
    def load_custom_sounds(self):
        try:
            with open('custom_sounds.pickle', 'rb') as f:
                self.custom_sounds = pickle.load(f)
        except FileNotFoundError:
            pass

    # Save custom sounds to file
    def save_custom_sounds(self):
        with open('custom_sounds.pickle', 'wb') as f:
            pickle.dump(self.custom_sounds, f)

    # Close the window
    def close_window(self):
        self.save_custom_sounds()  # Save custom sounds before closing
        if self.active:
            self.stop_voice_changer()  # Stop the voice changer
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = VoiceChangerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
