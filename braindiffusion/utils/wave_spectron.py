# This is brain waves to spectron with or without mel
# Migrated from https://github.com/huggingface/diffusers Under liscence
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import warnings
import numpy as np
import concurrent.futures
import librosa

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt

executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

try:
    import librosa  # noqa: E402

    _librosa_can_be_imported = True
    _import_error = ""
except Exception as e:
    _librosa_can_be_imported = False
    _import_error = (
        f"Cannot import librosa because {e}. Make sure to correctly install librosa to be able to install it."
    )


from PIL import Image  # noqa: E402


class Spectro(ConfigMixin, SchedulerMixin):
    """
    Parameters:
        x_res (`int`): x resolution of spectrogram (time)
        y_res (`int`): y resolution of spectrogram (frequency bins)
        sample_rate (`int`): sample rate of wave
        n_fft (`int`): number of Fast Fourier Transforms
        hop_length (`int`): hop length (a higher number is recommended for lower than 256 y_res)
        top_db (`int`): loudest in decibels
        n_iter (`int`): number of iterations for Griffin Linn mel inversion
    """

    config_name = "spectro_config.json"

    @register_to_config
    def __init__(
        self,
        x_res: int = 31,
        y_res: int = 64, # delta (0.5-4Hz), theta (4-8 Hz), alpha (8-14 Hz), beta (14-30Hz) and gamma (above 30Hz)
        sample_rate: int = 250, # 250hz for MI 500hz for reading
        n_fft: int = 100, # time window
        hop_length: int = 25, # time step
        top_db: int = 50, 
        n_iter: int = 32,
        eeg_channels : int = 22, 
    ):
        self.hop_length = hop_length
        self.sr = sample_rate
        self.n_fft = n_fft
        self.top_db = top_db
        self.n_iter = n_iter
        self.set_resolution(x_res, y_res)
        self.wave = None
        self.eeg_channels = eeg_channels
        self.batch_size = None

        if not _librosa_can_be_imported:
            raise ValueError(_import_error)

    def set_resolution(self, x_res: int, y_res: int):
        """Set resolution.

        Args:
            x_res (`int`): x resolution of spectrogram (time)
            y_res (`int`): y resolution of spectrogram (frequency bins)
        """
        self.x_res = x_res
        self.y_res = y_res
        self.n_mels = self.y_res
        self.slice_size = self.x_res * self.hop_length - 1

    def load_wave(self, wave_file: str = None, raw_wave: np.ndarray = None):
        """Load wave.

        Args:
            wave_file (`str`): must be a file on disk due to Librosa limitation or
            raw_wave (`np.ndarray`): wave as numpy array
        """
        if wave_file is not None:
            # self.wave, _ = librosa.load(wave_file, mono=True, sr=self.sr)
            raise ValueError(_import_error)
            print("load from file is not supported at this version")
        else:
            self.wave = raw_wave
        
        self.batch_size = self.wave.shape[0]
        assert self.eeg_channels == self.wave.shape[1]
        self.sample_len = self.wave.shape[2]

        # Pad with silence if necessary.
        if self.sample_len < self.x_res * self.hop_length:
            # print(len(self.wave))
            print("required lenght {}".format(self.x_res * self.hop_length))
            print("padding with shape {}".format(np.zeros((self.batch_size, self.eeg_channels, self.x_res * self.hop_length -(self.sample_len))).shape))
            # self.wave = np.concatenate([self.wave, np.zeros((self.batch_size, self.eeg_channels, self.x_res * self.hop_length - self.sample_len))])
            self.wave = np.concatenate([self.wave, np.zeros((self.batch_size, self.eeg_channels, self.x_res * self.hop_length - self.sample_len))], axis=2)

    def get_number_of_channels(self) -> int:
        """Get number of slices in wave.

        Returns:
            `int`: number of spectograms wave can be sliced into
        """
        return self.wave.shape[1]

    def get_wave_channel(self, slice: int = 0) -> np.ndarray:
        """Get slice of wave.

        Args:
            slice (`int`): slice number of wave (out of get_number_of_slices())

        Returns:
            `np.ndarray`: wave as numpy array
        """
        return self.wave[:slice:]

    def get_sample_rate(self) -> int:
        """Get sample rate:

        Returns:
            `int`: sample rate of wave
        """
        return self.sr
    
    def compute_mel_spec(self, channel):
            S = librosa.feature.melspectrogram(y=channel, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            log_S = librosa.power_to_db(S, ref=np.max, top_db=self.top_db)
            # log_S = S
            return log_S
    
    def wave_channelslice_to_latent_multitr(self, slice: int):
        """
        Convert a multi-channel EEG wave to a tensor of mel-spectrograms.

        Args:
            slice (`int`): slice number of wave to convert (out of get_number_of_slices())

        Returns:
        A tensor of mel-spectrograms with shape (bs, n_mels, T), where T is the number of frames
        # y_res x x_res x eeg_channels (y_res x x_res is the size of the lantent space)
        """
        # Load multi-channel audio data
        y = self.wave[:, slice, :]
        # Define a function to compute the mel-spectrogram for a single channel
        # Initialize a list to store the mel-spectrogram futures for each channel
        mel_spec_futures = []

        # Initialize a list to store the mel-spectrogram futures for each channel
        mel_spec_futures = []

        # Loop over each channel of the audio data
        for i in range(y.shape[0]):
            # Submit the compute_mel_spec function as a future and pass the i-th channel as an argument
            future = executor.submit(self.compute_mel_spec, y[i])
            # Append the future to the list
            mel_spec_futures.append(future)

        # Initialize a list to store the completed mel-spectrograms for each channel
        mel_specs = []

        # Loop over each mel-spectrogram future
        for future in concurrent.futures.as_completed(mel_spec_futures):
            # Get the completed mel-spectrogram from the future and append it to the list
            mel_spec = future.result()
            mel_specs.append(mel_spec)
            # print(mel_spec.shape)

        # Stack the mel-spectrograms along the third axis to create a tensor
        mel_specs = np.stack(mel_specs, axis=0).transpose(0, 2, 1)

        return mel_specs
    
    def wave_to_latent_multitr(self, slice=None):
        """
        Convert a multi-channel EEG wave to a tensor of mel-spectrograms.

        Args:
            slice (`int`): slice number of wave to convert (out of get_number_of_slices())

        Returns:
        A tensor of mel-spectrograms with shape (bs, n_mels, T), where T is the number of frames
        # y_res x x_res x eeg_channels (y_res x x_res is the size of the lantent space)
        """
        # Load multi-channel audio data
        y = self.wave.reshape(self.batch_size * self.eeg_channels, -1)
        # Define a function to compute the mel-spectrogram for a single channel
        # Initialize a list to store the mel-spectrogram futures for each channel
        mel_spec_futures = []

        # Initialize a list to store the mel-spectrogram futures for each channel
        mel_spec_futures = []

        # Loop over each channel of the audio data
        for i in range(y.shape[0]):
            # Submit the compute_mel_spec function as a future and pass the i-th channel as an argument
            future = executor.submit(self.compute_mel_spec, y[i])
            # Append the future to the list
            mel_spec_futures.append(future)

        # Initialize a list to store the completed mel-spectrograms for each channel
        mel_specs = []

        # Loop over each mel-spectrogram future
        for future in concurrent.futures.as_completed(mel_spec_futures):
            # Get the completed mel-spectrogram from the future and append it to the list
            mel_spec = future.result()
            mel_specs.append(mel_spec)
            # print(mel_spec.shape)

        # Stack the mel-spectrograms along the third axis to create a tensor
        mel_specs = np.stack(mel_specs, axis=0).reshape(self.batch_size, self.eeg_channels, self.n_mels, -1).transpose(0, 1, 3, 2)

        return mel_specs

    def wave_channelslice_to_latent(self, slice: int):
        """
        Convert a multi-channel EEG wave to a tensor of mel-spectrograms.

        Args:
            slice (`int`): slice number of wave to convert (out of get_number_of_slices())

        Returns:
        A tensor of mel-spectrograms with shape (bs, n_mels, T), where T is the number of frames
        # y_res x x_res x eeg_channels (y_res x x_res is the size of the lantent space)
        """
        # Load multi-channel audio data
        y = self.wave[:, slice, :]
        # Define a function to compute the mel-spectrogram for a single channel
        # Initialize a list to store the mel-spectrogram futures for each channel
        mel_specs = []

        for i in range(y.shape[0]):
            # compute the mel-spectrogram for the i-th channel
            S = librosa.feature.melspectrogram(y=y[i],  sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            # convert power spectrogram to dB-scaled spectrogram
            log_S = librosa.power_to_db(S, ref=np.max)
            # append the mel-spectrogram to the list
            # log_S = S
            mel_specs.append(log_S)

        # Stack the mel-spectrograms along the third axis to create a tensor
        mel_specs = np.stack(mel_specs, axis=0).transpose(0, 2, 1)

        return mel_specs
    
    def wave_to_latent(self, slice=None):
        """
        Convert a multi-channel EEG wave to a tensor of mel-spectrograms.

        Args:
            slice (`int`): slice number of wave to convert (out of get_number_of_slices())

        Returns:
        A tensor of mel-spectrograms with shape (bs, n_mels, T), where T is the number of frames
        # y_res x x_res x eeg_channels (y_res x x_res is the size of the lantent space)
        """
        # Load multi-channel audio data
        y = self.wave.reshape(self.batch_size * self.eeg_channels, -1)
        # Define a function to compute the mel-spectrogram for a single channel
        # Initialize a list to store the mel-spectrogram futures for each channel
        mel_specs = []

        for i in range(y.shape[0]):
            # compute the mel-spectrogram for the i-th channel
            S = librosa.feature.melspectrogram(y=y[i],  sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
            # convert power spectrogram to dB-scaled spectrogram
            log_S = librosa.power_to_db(S, ref=np.max)
            # append the mel-spectrogram to the list
            # log_S = S
            mel_specs.append(log_S)

        # Stack the mel-spectrograms along the third axis to create a tensor
        mel_specs = np.stack(mel_specs, axis=0).reshape(self.batch_size, self.eeg_channels, self.n_mels, -1).transpose(0, 1, 3, 2)

        return mel_specs
    
    def latent_to_wave(self, latent: np.ndarray) -> np.ndarray:
        """Converts spectrogram to wave.

        Args:
            image (`PIL Image`): x_res x y_res grayscale image

        Returns:
            wave (`np.ndarray`): raw wave
        """
        # bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
        latent = latent.transpose(0, 1, 3, 2)
        log_spectrogram = latent.reshape(self.batch_size * self.eeg_channels, self.n_mels, -1)
        waves = []
        for i in range(log_spectrogram.shape[0]):
            S = librosa.db_to_power(log_spectrogram[i])
            # S = log_spectrogram[i]
            wave = librosa.feature.inverse.mel_to_audio(
                S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter
            )
            waves.append(wave)

        return np.stack(waves, axis=0).reshape(self.batch_size, self.eeg_channels, -1)
    
    def single_latent_to_wave(self, latent: np.ndarray) -> np.ndarray:
        """Converts spectrogram to wave.

        Args:
            image (`PIL Image`): x_res x y_res grayscale image

        Returns:
            wave (`np.ndarray`): raw wave
        """
        # bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
        latent = latent.transpose(0, 2, 1)
        log_spectrogram = latent.reshape(self.eeg_channels, self.n_mels, -1)
        waves = []
        for i in range(log_spectrogram.shape[0]):
            S = librosa.db_to_power(log_spectrogram[i])
            # S = log_spectrogram[i]
            wave = librosa.feature.inverse.mel_to_audio(
                S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter
            )
            waves.append(wave)

        return np.stack(waves, axis=0).reshape(self.eeg_channels, -1)
    
    def compute_waves(self, log_spectrogram: np.ndarray) -> np.ndarray:
            S = librosa.db_to_power(log_spectrogram)
            # S = log_spectrogram
            wave = librosa.feature.inverse.mel_to_audio(
                S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter
            )
            # force rescaling need
            return wave
    
    def latent_to_wave_multitr(self, latent: np.ndarray) -> np.ndarray:
        """Converts spectrogram to wave.

        Args:
            image (`PIL Image`): x_res x y_res grayscale image

        Returns:
            wave (`np.ndarray`): raw wave
        """
        # bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
        latent = latent.transpose(0, 1, 3, 2)
        log_spectrogram = latent.reshape(self.batch_size * self.eeg_channels, self.n_mels, -1)

        future_waves = []
        for i in range(log_spectrogram.shape[0]):
            # Submit the compute_mel_spec function as a future and pass the i-th channel as an argument
            future = executor.submit(self.compute_waves, log_spectrogram[i])
            # Append the future to the list
            future_waves.append(future)
        waves = []
        for future in concurrent.futures.as_completed(future_waves):
            # Get the completed mel-spectrogram from the future and append it to the list
            wave = future.result()
            waves.append(wave)
      

        return np.stack(waves, axis=0).reshape(self.batch_size, self.eeg_channels, -1)
    
    def plot_spectrogram(self, latent, save_fig="./braindiffusion/visualization/spectrogram.png"):
        # Compute the spectrogram
        spectrogram = np.abs(latent).transpose(1,0)
        
        # Plot the spectrogram
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(spectrogram, cmap='hot', origin='lower')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Time')
        ax.set_title('Spectrogram')
        fig.colorbar(im, ax=ax)
        
        # Save the plot as an image
        if save_fig is not None:
            plt.savefig(save_fig)
        
        # Convert the plot to a PIL image and return it
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        return image
    
    
    def latent_to_image(self, latent: np.ndarray, bs=0, eeg_indx=0) -> Image.Image:
        """
        Convert slice of EEG wave (single channel) to spectrogram image and coloarize through PIL
        Params:
            latent (`np.ndarray`): latent space of EEG wave with shape (bs, eeg_channels, T, n_mels)
            bs (`int`): batch idex  
            eeg_indx (`int`): EEG channel index

        Returns:
            `PIL Image`: grayscale image of x_res x y_res
        """
        # bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
        latent = latent
        log_spectrogram = latent[bs, eeg_indx]
        image = self.plot_spectrogram(log_spectrogram)
        
        return image

    def wave_channelslice_to_image(self, slice: int) -> Image.Image:
        """Convert slice of EEG wave (single channel) to spectrogram image and coloarize through PIL
        Args:
            slice (`int`): slice number of wave to convert (out of get_number_of_slices())

        Returns:
            `PIL Image`: grayscale image of x_res x y_res
        """
        return self.latent_to_image(self.wave_channelslice_to_latent(slice))
        
    

if __name__ == "__main__":
    print("test start")
    sample_wave = np.random.rand(2, 22, 750)
    converter = Spectro()
    converter.load_wave(raw_wave=sample_wave)
    print(converter.wave.shape)
    print(converter.get_number_of_channels())
    print(converter.wave_channelslice_to_latent_multitr(0).shape)
    print(converter.wave_to_latent_multitr(0).shape)
    print(converter.wave_channelslice_to_latent(0).shape)
    print(converter.wave_to_latent(0).shape)
    inversed_wave = converter.latent_to_wave(converter.wave_to_latent(0))
    print(inversed_wave.shape)
    print(converter.latent_to_wave_multitr(converter.wave_to_latent_multitr(0)).shape)
    latent = converter.wave_to_latent_multitr(0)
    print(latent.shape)
    print(converter.plot_spectrogram(latent[0][0],save_fig='./test.png').size)
    print(sample_wave)
    # print(sample_wave-inversed_wave[:,:,:750])
    