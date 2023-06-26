import cv2
import numpy as np
import os
import argparse
import time

class ComparisonVisualizer:
    def __init__(self, color_images_path: str, depth_images_path: str, start_index: int = 0, fps_rate: int = 10, compare_type: str = 'depth') -> None:
        self.color_images_path = color_images_path
        self.depth_images_path = depth_images_path
        self.frequency = 1 / fps_rate
        self.start_index = start_index
        self.pause = False
        self.compare_type = compare_type

    def visualize(self) -> None:
        color_images = os.listdir(self.color_images_path)
        color_images.sort()
        depth_images = os.listdir(self.depth_images_path)
        depth_images.sort()

        if len(color_images) != len(depth_images):
            raise ValueError("Both directories should have the same number of images")

        self.start_index = min(self.start_index, len(color_images))

        color_images = color_images[self.start_index:]
        depth_images = depth_images[self.start_index:]
        
        for color_image_file, depth_image_file in zip(color_images, depth_images):
            color_image_path = os.path.join(self.color_images_path, color_image_file)
            depth_image_path = os.path.join(self.depth_images_path, depth_image_file)

            color_image = cv2.imread(color_image_path)
            #color_image = cv2.putText(color_image, color_image_file, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            depth_image = cv2.imread(depth_image_path)            
            #depth_image = cv2.putText(depth_image, depth_image_file, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #color_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            if self.compare_type == 'depth':
                depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
                # Inverts the greyscale image if the pixel is != 0 Just for visualization purposes
                depth_image=np.where(depth_image!=0, 255-depth_image, 0)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_HOT)


            if color_image.shape != depth_image.shape:
                color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]))

            color_depth_image = np.hstack((color_image, depth_image))

            color_depth_image = cv2.resize(color_depth_image, (1280, 360))


            cv2.imshow(f"Original vs {self.compare_type}", color_depth_image)
            time.sleep(self.frequency)
            key = cv2.waitKey(1)

            # pauses if space is pressed
            if key == ord(" "):
                self.pause = not self.pause
                if self.pause:
                    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize color and depth images")
    parser.add_argument("--color_images_path", type=str, help="Path to color images")
    parser.add_argument("--depth_images_path", type=str, help="Path to depth images")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of images")
    parser.add_argument("--fps_rate", type=int, default=10, help="FPS rate")
    parser.add_argument("--compare_type", type=str, default='depth', help="Type of images to compare with the original group. It can be depth or undistorted")
    args = parser.parse_args()

    color_images_path = args.color_images_path
    depth_images_path = args.depth_images_path
    start_index = args.start_index
    fps_rate = args.fps_rate
    compare_type = args.compare_type
    

    comparison_visualizer = ComparisonVisualizer(color_images_path, depth_images_path, start_index, fps_rate=fps_rate, compare_type=compare_type)
    comparison_visualizer.visualize()