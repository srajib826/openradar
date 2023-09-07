import mmwave as mm
from mmwave.dataloader import DCA1000
import numpy as np
dca = DCA1000()
frame_list = []
num_of_frames = 10



NUM_TX = 3 # tx order tx0,tx2,tx1, face to the board (left,right,upper) 
NUM_RX = 4

START_FREQ = 77 
ADC_START_TIME = 6 
FREQ_SLOPE = 60.012
ADC_SAMPLES = 256 # data samples per chirp
SAMPLE_RATE = 4400
RX_GAIN = 30 

IDLE_TIME = 7
RAMP_END_TIME = 65
#  2 for 1843
NUM_FRAMES = 0 
#  Set this to 0 to continuously stream data
LOOPS_PER_FRAME = 128 # num of chirp loop, one loop has three chirps
#PERIODICITY = 100 
# time for one chirp in ms  100ms == 10FPS
NUM_DOPPLER_BINS = LOOPS_PER_FRAME
NUM_RANGE_BINS = ADC_SAMPLES
NUM_ANGLE_BINS = 64
RANGE_RESOLUTION = (3e8 * SAMPLE_RATE * 1e3) / (2 * FREQ_SLOPE * 1e12 * ADC_SAMPLES)
MAX_RANGE = (300 * SAMPLE_RATE) / (2 * FREQ_SLOPE * 1e3)
DOPPLER_RESOLUTION = 3e8 / (2 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_DOPPLER_BINS * NUM_TX)
MAX_DOPPLER = 3e8 / (4 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_TX)

MMWAVE_RADAR_LOC=np.array([[0.146517, -3.030810, 1.0371905]]) # hard code the location of mmWave radar








def dopplerFFT(radar_cube):
    WindowesBin2D= radar_cube * np.reshape(np.hamming(128),(1, 1, -1, 1))
    dopplerFFTResult = np.fft.fft(WindowesBin2D, axis=2)
    dopplerFFTResult = np.fft.fft(dopplerFFTResult, axis=2)
    return dopplerFFTResult

def clutter_removal(input_val, axis=0):  #
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)
    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - mean
    return output_val.transpose(reordering)


def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64):  #
    assert num_tx > 2, "need a config for more than 2 TXs"
    num_detected_obj = virtual_ant.shape[1]
    azimuth_ant = virtual_ant[:2 * num_rx, :]
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant

    # Process azimuth information
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.abs(azimuth_fft), axis=0)
    peak_1 = np.zeros_like(k_max, dtype=np.complex_)
    for i in range(len(k_max)):
        peak_1[i] = azimuth_fft[k_max[i], i]

    k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
    wx = 2 * np.pi / fft_size * k_max
    x_vector = wx / np.pi

    # Zero pad elevation
    elevation_ant = virtual_ant[2 * num_rx:, :]
    elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    elevation_ant_padded[:num_rx, :] = elevation_ant

    # Process elevation information
    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)  # shape = (num_detected_obj, )
    peak_2 = np.zeros_like(elevation_max, dtype=np.complex_)
    for i in range(len(elevation_max)):
        peak_2[i] = elevation_fft[elevation_max[i], i]

    # Calculate elevation phase shift
    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    ypossible = 1 - x_vector ** 2 - z_vector ** 2
    y_vector = ypossible
    x_vector[ypossible < 0] = 0
    z_vector[ypossible < 0] = 0
    y_vector[ypossible < 0] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector

def frame2pointcloud(dopplerResult):
    dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0, 1))
    dopplerResultInDB = np.absolute(dopplerResultSumAllAntenna)

    

    cfarResult = np.zeros(dopplerResultInDB.shape, bool)
    top_size = 128
    energyThre128 = np.partition(dopplerResultInDB.ravel(), 128 * 256 - top_size - 1)[128 * 256 - top_size - 1]
    cfarResult[dopplerResultInDB > energyThre128] = True

    det_peaks_indices = np.argwhere(cfarResult == True)
    R = det_peaks_indices[:, 1].astype(np.float64)
    V = (det_peaks_indices[:, 0] - 128 // 2).astype(np.float64)
    
    R *= RANGE_RESOLUTION
    V *= DOPPLER_RESOLUTION
    energy = dopplerResultInDB[cfarResult == True]

    AOAInput = dopplerResult[:, :, cfarResult == True]
    AOAInput = AOAInput.reshape(12, -1)

    if AOAInput.shape[1] == 0:
        return np.array([]).reshape(6, 0)
    x_vec, y_vec, z_vec = naive_xyz(AOAInput)

    x, y, z = x_vec * R, y_vec * R, z_vec * R
    pointCloud = np.concatenate((x, y, z, V, energy, R))
    pointCloud = np.reshape(pointCloud, (6, -1))
    pointCloud = pointCloud[:, y_vec != 0]
    pointCloud = np.transpose(pointCloud, (1, 0))

    
    idx = np.argwhere(pointCloud[:, 4] > np.median(pointCloud[:, 4])).flatten()
    pointCloud = pointCloud[idx]

    
    return pointCloud

i = 0
while 1:
    adc_data = dca.read()
    range_fft = mm.dsp.range_processing(adc_data)
    range_fft = np.reshape(range_fft, (128, 3, 4, -1))
    range_fft = range_fft.transpose(1,2,0,3)
    #frame_list.append(range_fft)
    rangeResult = clutter_removal(range_fft, axis=2)
    

    dopplerResult = dopplerFFT(rangeResult)
    pointCloud = frame2pointcloud(dopplerResult)
    i+=1
    print(f'Frame Number : {i}, rangeresult shape : {rangeResult.shape}, doppler {dopplerResult.shape}, pointcloud: {pointCloud.shape}')
#frame_list = np.array(frame_list)
#print(frame_list.shape)


