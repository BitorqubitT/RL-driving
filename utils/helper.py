import pandas as pd
import time
import torch
import math

"""
    Contains helper functions for training.
"""

def calc_mean(scores):
    scores = torch.tensor(scores, dtype=torch.float)
    # Take 100 episode averages and plot them too
    if len(scores) >= 100:
        means = scores.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        return means.numpy()[-1]
    return 0

def rotate_point(px, py, angle):
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    x_new = px * cos_angle - py * sin_angle
    y_new = px * sin_angle + py * cos_angle
    return x_new, y_new

def calculate_angle(source, target):
    velocity = (target[0] - source[0], target[1] - source[1])
    angle = math.atan2(velocity[1], velocity[0])
    return angle

def is_point_on_line2(p1, p2, p):
        #print(p1, p2, p)
        # Calculate the cross product of vectors (p1 -> p) and (p1 -> p2)
        dxc = p[0] - p1[0]
        dyc = p[1] - p1[1]
        dxl = p2[0] - p1[0]
        dyl = p2[1] - p1[1]
        cross = dxc * dyl - dyc * dxl

        # If the cross product is zero, the point is on the line
        # Shouldnt this be 0?
        if cross >= 1:
            return False

        # Check if the point is within the bounds of the line segment
        if abs(dxl) >= abs(dyl):
            return p1[0] <= p[0] <= p2[0] if dxl > 0 else p2[0] <= p[0] <= p1[0]
        else:
            return p1[1] <= p[1] <= p2[1] if dyl > 0 else p2[1] <= p[1] <= p1[1]
        
def is_point_on_line(p1, p2, p, tolerance):
    #print(p1, p2, p)
    x1, y1 = p1
    x2, y2 = p2
    x, y = p
    # Calculate the slope of the line formed by (x1, y1) and (x2, y2)
    if x2 - x1 == 0:  # Vertical line
        return abs(x - x1) <= tolerance
    slope = (y2 - y1) / (x2 - x1)
    
    # Calculate the y-intercept of the line
    intercept = y1 - slope * x1
    
    # Check if the point (x, y) satisfies the line equation y = slope * x + intercept
    return abs(y - (slope * x + intercept)) <= tolerance
