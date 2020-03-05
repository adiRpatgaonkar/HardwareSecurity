from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import string
from tqdm import trange
SHRINK_SCALE = 20
NUM_CLICKS_TRAIN = 100
CLICK_DETECT_FACTOR = 10
CLICK_LEN = 3000
NUM_LETTERS = 26
DEBUG = False


class KeyExtracter:
    def __init__(self, file_name, num_clicks=NUM_CLICKS_TRAIN):
        self.file_name = file_name
        self.shrink_scale = SHRINK_SCALE
        self.num_clicks = num_clicks
        self.click_len = CLICK_LEN  # 3000 * 1/44kHz = 0.07 second
        self.click_detect_factor = CLICK_DETECT_FACTOR

    def _read_file(self):
        self.freq, self.arr = wavfile.read(self.file_name)
        self.arr_len = len(self.arr)

    def _identify_clicks(self):
        abs_arr = np.abs(self.arr)
        shrinked_abs_arr_len = self.arr_len//self.shrink_scale
        shrinked_abs_arr = np.zeros((shrinked_abs_arr_len,), dtype=np.int32)
        for i in range(shrinked_abs_arr_len):
            shrinked_abs_arr[i] = np.average(
                abs_arr[i*self.shrink_scale:
                        i*self.shrink_scale+self.shrink_scale])
        abs_arr_avg = np.average(abs_arr)
        self.click_starts = np.zeros((self.num_clicks), dtype=np.int32)
        click_idx = 0
        arr_idx = 0
        while click_idx < self.num_clicks:
            if shrinked_abs_arr[arr_idx] > self.click_detect_factor * abs_arr_avg:
                self.click_starts[click_idx] = arr_idx * \
                    self.shrink_scale - 5 * self.shrink_scale
                arr_idx += 2 * self.click_len//self.shrink_scale
                click_idx += 1
            else:
                arr_idx += 1

    def _process_clicks(self):
        avg = np.zeros((self.click_len), dtype=np.int32)

        self.arrs = np.zeros(
            (self.num_clicks, self.click_len), dtype=np.int32)
        for i in range(self.num_clicks):
            start = self.click_starts[i]
            seg = self.arr[start:start+self.click_len]
            avg += seg
            self.arrs[i] = seg
            if DEBUG:
                plt.plot(self.arrs[i])
                plt.savefig(self.file_name + str(i) + ".png")
                plt.clf()
        plt.plot(avg/self.num_clicks)
        plt.savefig(self.file_name + ".png")
        plt.clf()

    def run(self):
        self._read_file()
        self._identify_clicks()
        self._process_clicks()


class Data:
    def __init__(self):
        self.train_data = np.zeros(
            (NUM_LETTERS*NUM_CLICKS_TRAIN, CLICK_LEN), dtype=np.int32)
        self.train_label = np.zeros(
            (NUM_LETTERS*NUM_CLICKS_TRAIN), dtype=np.int32)

    def run(self):
        for i in trange(NUM_LETTERS):
            char = string.ascii_lowercase[i]
            key = KeyExtracter("./data/{}.wav".format(char))
            key.run()
            self.train_data[i * NUM_CLICKS_TRAIN:
                            (i+1)*NUM_CLICKS_TRAIN] = key.arrs
            self.train_label[i * NUM_CLICKS_TRAIN:
                             (i+1)*NUM_CLICKS_TRAIN] = i

    def save(self):
        np.savez("train", data=self.train_data, label=self.train_label)


if __name__ == "__main__":
    data = Data()
    data.run()
    data.save()
