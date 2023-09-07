mport mmwave as mm
from mmwave.dataloader import DCA1000
import numpy as np
dca = DCA1000()
frame_list = []
num_of_frames = 10
for i in range(num_of_frames):
    adc_data = dca.read()
    radar_cube = mm.dsp.range_processing(adc_data)
    radar_cube = np.reshape(radar_cube, (128, 3, 4, -1))
    radar_cube = radar_cube.transpose(1,2,0,3)
    frame_list.append(radar_cube)
frame_list = np.array(frame_list)
print(frame_list.shape)

