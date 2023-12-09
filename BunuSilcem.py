import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

DATA_DIR = './data'


eller = mp.solutions.hands
cizim = mp.solutions.drawing_utils
cizim_teknigi = mp.solutions.drawing_styles

