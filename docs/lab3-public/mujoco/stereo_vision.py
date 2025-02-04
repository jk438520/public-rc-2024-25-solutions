import mujoco
from PIL import Image
import cv2
import numpy as np

# distance between two cameras
baseline = 0.1

sphere_size = 0.01
box1_size = 0.05
box2_size = 0.02

resolution = (1280, 1280)

img1 = cv2.imread('left.png')
img2 = cv2.imread('right.png')

#TODO: find the positions of the objects and reconstruct the images
def get_centroid(contour):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx, cy


def find_contour_of_range(img, lowe, upper):
    mask = cv2.inRange(img, lowe, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    with_cnt = cv2.drawContours(np.copy(img), contours, -1, (0,255,0), 3)
    
    cx, cy = get_centroid(contours[0])
    color = img[cy, cx]
    
    with_cnt = cv2.circle(with_cnt, (cx, cy), 3, color=(255, 255, 0))
    cv2.imwrite(f"contours{color}.png", with_cnt)
    
    return contours
    


red_lower = np.array([00, 00, 0x5c])
red_upper = np.array([0x2d, 0x2d, 0xab])

green_lower = np.array([0x02, 0x82, 0x02]) - 2
green_upper = green_lower + 25

blue_lower = np.array([0x97, 0x17, 0x17]) - 20
blue_upper = blue_lower + 35


f = 640 / np.tan(np.pi / 4)

def xyz(ul, vl, ur, vr):
    ox = 640
    oy = 640
    x = baseline * (ul - ox) / (ul - ur)
    y = baseline * (vl - oy) / (ul - ur)
    z = baseline * f / (ul - ur)
    return x, y, z

def match_centroid(img1, img2, lower_bound, upper_bound):
    countur1 = find_contour_of_range(img1, lower_bound, upper_bound)
    countur2 = find_contour_of_range(img2, lower_bound, upper_bound)

    center1x, center1y = get_centroid(countur1[0])
    center2x, center2y = get_centroid(countur2[0])

    print(f"{(center1x, center1y)=}")
    print(f"{(center2x, center2y)=}")
    
    return  xyz(center1x, center1y, center2x, center2y)

red_ball_pos = np.array(match_centroid(img1, img2, red_lower, red_upper))

green_top_pos = np.array(match_centroid(img1, img2, green_lower, green_upper))

blue_top_pos = np.array(match_centroid(img1, img2, blue_lower, blue_upper))

green_size = 0.1
blue_size = 0.04

def push_and_swap(v):
    
    return np.array([v[0], -v[1], 1-v[2]])


green_mid_pose = green_top_pos + np.array([0, 0, green_size/2])
blue_mid_pose = blue_top_pos + np.array([0, 0, blue_size/2])

red_ball_pos = push_and_swap(red_ball_pos)
green_mid_pose = push_and_swap(green_mid_pose)
blue_mid_pose = push_and_swap(blue_mid_pose)

# red sphere
pos_sphere = red_ball_pos

# green box
pos_box1 = green_mid_pose

# blue box
pos_box2 = blue_mid_pose

xml_string =f"""\
<mujoco model="simple_scene">
      <visual>
     <global offwidth="{resolution[0]}" offheight="{resolution[1]}"/>
  </visual>
    <asset>
        <texture name="plane_texture" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="512" height="512"/>
        <material name="plane_material" texture="plane_texture" texrepeat="5 5" reflectance="0."/>
    </asset>

    <worldbody>
        <geom type="plane" size="1 1 0.1" material="plane_material"/>

        <body pos="{pos_sphere[0]} {pos_sphere[1]} {pos_sphere[2]}">
            <geom type="sphere" size="{sphere_size}" rgba="1 0 0 1"/>
        </body>

        <body pos="{pos_box1[0]} {pos_box1[1]} {pos_box1[2]}">
            <geom type="box" size="0.05 0.05 0.05" rgba="0 1 0 1"/>
        </body>

        <body pos="{pos_box2[0]} {pos_box2[1]} {pos_box2[2]} ">
            <geom type="box" size="0.02 0.02 0.02" rgba="0 0 1 1"/>
        </body>

        <!-- Cameras -->
        <camera name="camera1" pos="0 0 1" fovy="90"/>
        <camera name="camera2" pos="{baseline} 0 1" fovy="90"/>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_string)
renderer = mujoco.Renderer(model, resolution[0], resolution[1])

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer.update_scene(data, camera="camera1")
img1_r = Image.fromarray(renderer.render())
img1_r.save("reconstruct_left.png")

renderer.update_scene(data, camera="camera2")
img2_r = Image.fromarray(renderer.render())
img2_r.save("reconstruct_right.png")
