# display.py
from config_loader import load_config
import pygame, sys

config = load_config()
z_offset = config["geometry"]["z_offset"]
H = config["geometry"]["H"]
scale_factor = config["geometry"]["scale_factor"]

def _calSizeRatio(target):
    x, y, z = target
    distance = (x**2 + y**2 + (z-H)**2 - z_offset**2)**0.5
    return standard / distance
