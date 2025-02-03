import mujoco
from PIL import Image
import cv2
import numpy as np

xml_path = "world.xml"
model = mujoco.MjModel.from_xml_string(open(xml_path).read())
renderer = mujoco.Renderer(model, 1024, 1024)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera="camera1")

img = Image.fromarray(renderer.render())
img.save("red_ball.png")
cv2im = np.array(img)
cv2.imwrite("red_ball_cv2.png", cv2im)

lower = np.array([0x4a,00,00])

upper = np.array([0x95,0x32, 0x32])

mask = cv2.inRange(cv2im, lower, upper)
cv2.imwrite("mask.png", mask)