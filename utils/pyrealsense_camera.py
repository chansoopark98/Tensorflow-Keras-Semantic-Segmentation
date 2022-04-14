import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def connect(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
            
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        # config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, self.fps)

        self.cfg = self.pipeline.start(config)

        # Determine intrinsics
        s = self.cfg.get_device().query_sensors()[1]
        s.set_option(rs.option.enable_auto_exposure, False)
        s.set_option(rs.option.exposure, 10)
    
    
        rgb_profile = self.cfg.get_stream(rs.stream.color)
    
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = self.cfg.get_device().first_depth_sensor().get_depth_scale() # l515 : 0.00025 d435 : 0.001
        # self.scale *= 4.

    def get_image_bundle(self):
        align_to = rs.stream.color
        align = rs.align(align_to)

        frames = self.pipeline.wait_for_frames()
        
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        return {
            'rgb': color_image,
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()


if __name__ == '__main__':
    cam = RealSenseCamera(device_id='f1231507') #0003b661b825 # f0350818 # f1231507 # 0003b9fa147c
    cam.connect()
    while True:
        cam.plot_image_bundle()
