import mmwave as mm
from mmwave.dataloader import DCA1000
import numpy as np
dca = DCA1000()
frame_list = []
num_of_frames = 10
def dopplerFFT(radar_cube, frameConfig):
    windowedBins2D = radar_cube * np.reshape(np.hamming(frameConfig.numLoopsPerFrame),(1, 1, -1, 1))
    dopplerFFTResult = np.fft.fft(WindowesBin2D, axis=2)
    dopplerFFTResult = np.fft.fft(dopplerFFTresult, axes=2)
    return dopplerFFTResult

for i in range(num_of_frames):
    adc_data = dca.read()
    radar_cube = mm.dsp.range_processing(adc_data)
    radar_cube = np.reshape(radar_cube, (128, 3, 4, -1))
    radar_cube = radar_cube.transpose(1,2,0,3)
    frame_list.append(radar_cube)
    dopplerFFT(radar_cube, (128, 3, 4, -1))
frame_list = np.array(frame_list)
print(frame_list.shape)


