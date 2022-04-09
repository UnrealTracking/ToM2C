Cam_Pose = [[-742, 706, -62.842588558231455], [-843, 69, -26.590794532324466], [510, 703, -135.84636503921902],
            [466, -609, 153.13035548432399]]
Target_Pose = [[407.90650859, -716.624028],
               [-64.83188835, -233.64760113],
               [-980.29575616, 201.18355808],
               [-493.24174167, 655.69319226],
               [-571.57383471, -673.35637078]]
Target_camera_dict = {0: [], 1: [3], 2: [], 3: [], 4: [1]}
Camera_target_dict = {0: [], 1: [4], 2: [], 3: [1]}
Distance = [[1829.24686786, 1158.22893495, 558.23338079, 253.79410157, 1389.84498251],
            [1477.15002847, 834.94980715, 190.58493563, 683.03714475, 790.42086539],
            [1423.29036457, 1098.97244214, 1572.51428681, 1004.35647371, 1750.47388421],
            [122.3020243, 650.13223042, 1657.7601793, 1587.3227742, 1039.56779718]]
reward = [0.4]
goals4cam = [[1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1]]

import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
visual_distance = 100


def render(camera_pos, target_pos,
           obstacle_pos=None, goal=None, comm_edges=None, obstacle_radius=None, save=False, visible=None):
    camera_pos = np.array(camera_pos)
    target_pos = np.array(target_pos)

    camera_pos[:, :2] /= 1000.0
    target_pos[:, :2] /= 1000.0

    length = 600
    area_length = 1  # for random cam loc
    target_pos[:, :2] = (target_pos[:, :2] + 1) / 2
    camera_pos[:, :2] = (camera_pos[:, :2] + 1) / 2

    img = np.zeros((length + 1, length + 1, 3)) + 255
    num_cam = len(camera_pos)
    camera_position = [camera_pos[i][:2] for i in range(num_cam)]
    camera_position = length * (1 - np.array(camera_position) / area_length) / 2
    abs_angles = [camera_pos[i][2] * -1 for i in range(num_cam)]

    num_target = len(target_pos)
    target_position = [target_pos[i][:2] for i in range(num_target)]
    target_position = length * (1 - np.array(target_position) / area_length) / 2

    if obstacle_pos.shape != () :
        num_obstacle = len(obstacle_pos)
        obstacle_pos = np.array(obstacle_pos)
        obstacle_pos[:, :2] /= 1000.0
        obstacle_pos[:, :2] = (obstacle_pos[:, :2] + 1) / 2
        obstacle_position = [obstacle_pos[i][:2] for i in range(num_obstacle)]
        obstacle_position = length * (1 - np.array(obstacle_position) / area_length) / 2

    fig = plt.figure(0)
    plt.cla()
    plt.imshow(img.astype(np.uint8))

    # get camera's view space positions
    visua_len = 100  # length of arrow
    L = 120  # length of arrow
    ax = plt.gca()
    # obstacle
    if obstacle_pos.shape != () :
        for i in range(num_obstacle):
            disk_obs = plt.Circle((obstacle_position[i][0] + visua_len, obstacle_position[i][1] + visua_len), obstacle_radius[i] * L/800, color='grey', fill=True)
            ax.add_artist(disk_obs)
            plt.annotate(str(i + 1), xy=(obstacle_position[i][0] + visua_len, obstacle_position[i][1] + visua_len),
                         xytext=(obstacle_position[i][0] + visua_len, obstacle_position[i][1] + visua_len), fontsize=10,
                         color='black')

    for i in range(num_cam):
        # drawing the visible area of a camera
        # dash-circle
        r = L
        a, b = np.array(camera_position[i]) + visua_len
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = a + r * np.cos(theta)
        y = b + r * np.sin(theta)
        plt.plot(x, y, linestyle=' ',
                 linewidth=1,
                 color='steelblue',
                 dashes=(6, 5.),
                 dash_capstyle='round',
                 alpha=0.9)

        # fill circle
        disk1 = plt.Circle((a, b), r, color='steelblue', fill=True, alpha=0.05)
        ax.add_artist(disk1)
        #

    for i in range(num_cam):
        theta = abs_angles[i]  # -90
        theta -= 90
        the1 = theta - 45
        the2 = theta + 45

        a = camera_position[i][0] + visua_len
        b = camera_position[i][1] + visua_len
        wedge = mpatches.Wedge((a, b), L, the1*-1, the2*-1+180, color='green', alpha=0.2)  # drawing the current sector that the camera is monitoring
        # print(i, the1*-1, the2*-1)
        ax.add_artist(wedge)

        disk1 = plt.Circle((camera_position[i][0] + visua_len, camera_position[i][1] + visua_len), 4, color='navy', fill=True) # drawing the camera
        ax.add_artist(disk1)
        plt.annotate(str(i + 1), xy=(camera_position[i][0] + visua_len, camera_position[i][1] + visua_len),
                     xytext=(camera_position[i][0] + visua_len, camera_position[i][1] + visua_len), fontsize=10,
                     color='black')
    
    # draw the communication edges
    if comm_edges is not None:                 
        edges = (comm_edges.numpy()).reshape(num_cam, num_cam)
        # print the edge matrix and the sum of edges
        edge_cnt = np.sum(edges)
        plt.text(600,470, 'Total {} Comm Edges :'.format(edge_cnt), color="black")
        for i in range(num_cam):
            for j in range(num_cam):
                edge = edges[i][j]
                if edge:
                    x,y = np.array(camera_position[i]) + visua_len
                    x_target, y_target = np.array(camera_position[j]) + visua_len
                    dx,dy = x_target-x, y_target-y
                    ax.arrow(x, y, dx, dy, head_width=15, head_length=15, fc='y', ec='k')
            plt.text(600, 500 + i * 30, str(edges[i]))

    plt.text(5, 5, '{} sensors & {} targets'.format(num_cam, num_target), color="black")

    for i in range(num_target):
        c = 'firebrick'
        for j in range(num_cam):
            if visible[j][i]:
                c = 'yellow'

        plt.plot(target_position[i][0] + visua_len, target_position[i][1] + visua_len, color=c,
                 marker="o")
        plt.annotate(str(i + 1), xy=(target_position[i][0] + visua_len, target_position[i][1] + visua_len),
                     xytext=(target_position[i][0] + visua_len, target_position[i][1] + visua_len), fontsize=10,
                     color='maroon')

    if goal is not None:
        plt.text(400, 470, 'Goals:')
        for i in range(len(goal)):
            tmp = np.zeros(len(goal[i]))
            tmp[goal[i] > 0.5] = 1
            plt.text(400, 500 + i * 30, str(tmp))

    plt.axis('off')
    # plt.show()
    if save:
        file_path = '../demo/img'
        file_name = '{}.jpg'.format(datetime.now())
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        plt.savefig(os.path.join(file_path, file_name))
    plt.pause(0.01)


if __name__ == '__main__':
    render(Cam_Pose, Target_Pose, reward, np.array(goals4cam))