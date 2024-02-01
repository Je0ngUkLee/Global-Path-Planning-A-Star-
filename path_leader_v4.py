#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from warnings import warn
# from PIL import Image

import rospy
import signal
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def signal_handler(signal, frame):
  print('\nYou pressed Ctrl+C!')
  sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)



class Astar:
  def __init__(self, grid_size, robot_radius, start, goal, show_animation, scene_num):
    self.grid_size = grid_size
    self.robot_radius = robot_radius
    self.allow_diagonal_movement = True
    self.show_animation = show_animation
    self.scene_num = scene_num

    self.x_width = 100
    self.y_width = 100
    self.maze = np.zeros((int(self.x_width / self.grid_size), int(self.y_width / self.grid_size)), dtype = int)

    self.start_position = start
    self.goal_position = goal

    self.start_node = Node(None, self.calc_grid_pos(self.start_position))
    self.goal_node = Node(None, self.calc_grid_pos(self.goal_position))

    self.open_list = []
    self.closed_list = []

    self.open_list.append(self.start_node)

    self.adjacent_squares = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    if self.allow_diagonal_movement:
      self.adjacent_squares += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            
    self.outer_iteration = 0
    self.max_iteration = (len(self.maze[len(self.maze) - 1]) // 2) ** 2

    self.obstacles = []  # Obstacle(x, y, r)
    self.obs_grid = []
    
    self.set_scenario()
    
    
  def set_scenario(self):
    # map 1
    # for i in range(0, 100):
    #   self.obstacles.append(Obstacle(i, -50, 1))
    #   self.obstacles.append(Obstacle(i, 50, 1))
    # for i in range(-50, 50):
    #   self.obstacles.append(Obstacle(0, i, 1))
    #   self.obstacles.append(Obstacle(100, i, 1))
    
    # map2
    for i in range(0, 50):
      self.obstacles.append(Obstacle(i, 6, 1))
      self.obstacles.append(Obstacle(i, -6, 1))
    for i in range(-6, 7):
      self.obstacles.append(Obstacle(0, i, 1))
      self.obstacles.append(Obstacle(50, i, 1))
      
    # 2024-01-26 [RaL Scene1 _ obs grid _ map2]
    for i in range(0, 51):
      self.obs_grid.append(Obstacle(i, 6 + 0.5, 1))
      self.obs_grid.append(Obstacle(i, -6 - 0.5, 1))
      
    for i in range(-6, 8):
      self.obs_grid.append(Obstacle(0 - 0.5, i - 0.5, 1))
      self.obs_grid.append(Obstacle(50 + 0.5, i - 0.5, 1))
      
    if self.scene_num == 1:
      # 2023-10-11 [RaL Scene1]
      for i in range(10, 12):
        for j in range(2, 7):
          self.obstacles.append(Obstacle(i, j, 1))
        for j in range(-6, -1):
          self.obstacles.append(Obstacle(i, j, 1))
        
      for i in range(20, 22):
        for j in range(0, 7):
          self.obstacles.append(Obstacle(i, j, 1))
        for j in range(-6, -3):
          self.obstacles.append(Obstacle(i, j, 1))
          
      for i in range(30, 32):
        for j in range(4, 7):
          self.obstacles.append(Obstacle(i, j, 1))
        for j in range(-6, 1):
          self.obstacles.append(Obstacle(i, j, 1))
          
      # 2024-01-26 [RaL Scene1 _ obs grid]
      for i in range(2, 6):
        self.obs_grid.append(Obstacle(10.5, i + 0.5, 1))
      for i in range(-5, -1):
        self.obs_grid.append(Obstacle(10.5, i - 0.5, 1))
        
      for i in range(0, 6):
        self.obs_grid.append(Obstacle(20.5, i + 0.5, 1))
      for i in range(-5, -3):
        self.obs_grid.append(Obstacle(20.5, i - 0.5, 1))
        
      for i in range(4, 6):
        self.obs_grid.append(Obstacle(30.5, i + 0.5, 1))
      for i in range(-5, 1):
        self.obs_grid.append(Obstacle(30.5, i - 0.5, 1))
      
    elif self.scene_num == 2:
      # 2024-01-17 [RaL Scene2]
      for i in range(13, 21):
        for j in range(2, 6):
          self.obstacles.append(Obstacle(i, j, 1))
        for j in range(-5, -1):
          self.obstacles.append(Obstacle(i, j, 1))
          
      for i in range(28, 36):
        for j in range(2, 6):
          self.obstacles.append(Obstacle(i, j, 1))
        for j in range(-5, -1):
          self.obstacles.append(Obstacle(i, j, 1))
    
      # 2024-01-26 [RaL Scene2 _ obs grid]
      for i in range(13, 20):
        for j in range(2, 6):
          self.obs_grid.append(Obstacle(i + 0.5, j + 0.5, 1))
        for j in range(-5, -1):
          self.obs_grid.append(Obstacle(i + 0.5, j - 0.5, 1))
          
      for i in range(28, 35):
        for j in range(2, 6):
          self.obs_grid.append(Obstacle(i + 0.5, j + 0.5, 1))
        for j in range(-5, -1):
          self.obs_grid.append(Obstacle(i + 0.5, j - 0.5, 1))
      
    self.update_obstacles()
    

  def calc_grid_pos(self, node):
    return int(((len(self.maze) / 2 * self.grid_size) - node[1]) / self.grid_size), int(node[0] / self.grid_size)
  
  
  def print_figure(self, map, title):
    cmap = plt.get_cmap('gray')
    cmap.set_bad(color = 'black')
  
    extent = [0, self.maze.shape[1]/2, 0, self.maze.shape[0]/2]  # x_min, x_max, y_min, y_max
    
    plt.figure(figsize = (10, 10))
    plt.imshow(1 - map, cmap = cmap, origin = 'upper', extent = extent, vmin = 0, vmax = 1)
    plt.title(title)
    plt.show()
    
    
  def erode_obstacles(self):
    self.kernel_size = (2, 2)
    
    self.save_map_as_image(self.maze, 'original_map_image.png')
    if self.show_animation:
      self.print_figure(self.maze, 'Original_map')
    
    # 장애물 erode
    erode_kernel = np.ones(self.kernel_size, dtype = np.uint8)
    self.maze = self.erode(self.maze, erode_kernel)
    
    self.save_map_as_image(self.maze, 'erode_map_image.png')
    
    if self.show_animation:
      self.print_figure(self.maze, 'Eroded_map')
      
    
  def dilate_obstacles(self):
    # 장애물 dilate
    dilate_kernel = np.ones(self.kernel_size, dtype = np.uint8)
    self.maze = self.dilate(self.maze, dilate_kernel)
    
    self.save_map_as_image(self.maze, 'dilate_map_image.png')
    
    if self.show_animation:
      self.print_figure(self.maze, 'Dilated_map')
    
    
  @staticmethod
  def erode(map, kernel):
    return np.minimum.reduce([
      np.roll(np.roll(map, i, axis = 0), j, axis = 1) for i in range(-kernel.shape[0] // 2, kernel.shape[0] // 2 + 1)
                                                        for j in range(-kernel.shape[1] // 2, kernel.shape[1] // 2 + 1)
    ])
    
  @staticmethod
  def dilate(map, kernel):
    return np.maximum.reduce([
      np.roll(np.roll(map, i, axis = 0), j, axis = 1) for i in range(-kernel.shape[0] // 2, kernel.shape[0] // 2 + 1)
                                                        for j in range(-kernel.shape[1] // 2, kernel.shape[1] // 2 + 1)
    ])
    
  def save_map_as_image(self, map, filename):
    # map = map * 255
    map = (map != 9).astype(np.uint8) * 255
    # map = map.astype(np.uint8)
    map_image = Image.fromarray(map, 'L')
    map_image.save(filename)
    # map_image.show()

  def update_obstacles(self):
    for obs in self.obstacles:
      obs.pos = self.calc_grid_pos((obs.x, obs.y))
      if obs.r > self.grid_size:
        k = (obs.r - self.grid_size) / self.grid_size
        for i in range(-int(k), int(k) + 1):
          for j in range(-int(k), int(k) + 1):
            if not self.check_range((obs.pos[0] + i, obs.pos[1] + j)):
              self.maze[int(obs.pos[0]) + i][int(obs.pos[1]) + j] = 9
      if not self.check_range(obs.pos):
        self.maze[int(obs.pos[0])][int(obs.pos[1])] = 9

  def check_range(self, node):
    return (
      node[0] > (len(self.maze) - 1)
      or node[0] < 0
      or node[1] > (len(self.maze[len(self.maze) - 1]) - 1)
      or node[1] < 0
    )

  def astar_condition(self):
    return len(self.open_list) > 0

  def get_the_current_node(self):
    self.current_node = self.open_list[0]
    self.current_index = 0
    for index, item in enumerate(self.open_list):
      if item.F < self.current_node.F:
        self.current_node = item
        self.current_index = index
        
      elif item.F == self.current_node.F:
        if item.H < self.current_node.H:
          self.current_node = item
          self.current_index = index
          
    if self.outer_iteration > self.max_iteration:
      warn("giving up on pathfinding too many iterations")
      return return_path(self.current_node)
    
    self.open_list.pop(self.current_index)
    self.closed_list.append(self.current_node)

  def check_find_the_goal(self):
    # print(self.current_node.position)
    # print(self.goal_node.position)
    # print('\n')
    if self.current_node.position == self.goal_node.position:
      return return_path(self.current_node)

  def generate_children(self):
    self.children = []
    for new_position in self.adjacent_squares:
      self.node_position = (
        self.current_node.position[0] + new_position[0],
        self.current_node.position[1] + new_position[1],
      )
      if self.check_range(self.node_position):
        continue
      if self.maze[int(self.node_position[0])][int(self.node_position[1])] != 0:
        continue
      if self.check_radius():
        continue
      self.new_node = Node(self.current_node, self.node_position)
      self.children.append(self.new_node)

  # def check_radius(self):
  #   for i in range(int(self.robot_radius)):
  #     if (self.maze[int(self.node_position[0] + 1 + i)][int(self.node_position[1])] != 0):
  #       return True
  #     if (self.maze[int(self.node_position[0] - 1 - i)][int(self.node_position[1])] != 0):
  #       return True
  #     if (self.maze[int(self.node_position[0])][int(self.node_position[1] + 1 + i)] != 0):
  #       return True
  #     if (self.maze[int(self.node_position[0])][int(self.node_position[1] - 1 - i)] != 0):
  #       return True
  #   return False
  
  def check_radius(self):
    for idx, obs in enumerate(self.obstacles):
      node_x = self.node_position[1] * self.grid_size
      node_y = ((len(self.maze) / 2 * self.grid_size) - self.node_position[0] * self.grid_size)
      
      dist = math.hypot(node_x - obs.x, node_y - obs.y)
      
      if dist < self.robot_radius:
        if obs.x == 10 and node_x == 9 and node_y == 0:
          print('\n')
          print('node_grid', node_x, node_y)
          print(dist)
          print('obstacle', obs.x, obs.y)
        return True
      
    return False
      
      
      # result = np.ones((len(path_node), 2))

      # for i in range(len(path_node)):
      #   result[i][1] = ((len(astar.maze) / 2 * astar.grid_size) - path_node[i][0] * astar.grid_size)
      #   result[i][0] = path_node[i][1] * astar.grid_size

      # rx, ry = [], []

      # for i in range(len(result)):
      #   rx.append(result[i][0])
      #   ry.append(result[i][1])
      

  def loop_through_children(self):
    self.heuristic_weight = 1.0
    
    for child in self.children:
      if len([closed_child for closed_child in self.closed_list if closed_child == child]) > 0:
        continue
      if child.position[0] == self.current_node.position[0] or child.position[1] == self.current_node.position[1]:
        child.G = self.current_node.G + 10
      else:
        child.G = self.current_node.G + 14
      
      child.H = (abs(child.position[0] - self.goal_node.position[0]) + abs(child.position[1] - self.goal_node.position[1])) * 10
      
      child.F = child.G + self.heuristic_weight * child.H
      
      same_ = False
      for idx, open_node in enumerate(self.open_list):
        if open_node.position == child.position:
          same_ = True
          same_idx = idx
          
          if open_node.F > child.F:
            self.open_list.pop(same_idx)
            self.open_list.append(child)
            
      if same_ == False:
        self.open_list.append(child)


class Node:
  def __init__(self, parent=None, position=None):
    self.parent = parent
    self.position = position
    self.G = 0
    self.H = 0
    self.T = 0
    self.A = 0
    self.F = 0

  def __eq__(self, other):
    return self.position == other.position

  def __str__(self):
    return str(self.position)


class Obstacle:
  def __init__(self, x, y, r):
    self.x = x
    self.y = y
    self.r = r
    self.pos = (0, 0)


class ROS:
  def __init__(self):
    rospy.init_node('Reference_path_node')
    self.global_path_pub = rospy.Publisher('reference_path', Path, queue_size=1)

  def path_publish(self, x, y):
    pub_msg = Path()
    pub_msg.header.frame_id = 'map'
    pub_msg.header.stamp = rospy.Time.now()
    
    for i in range(len(x)):
      pose_stamped = PoseStamped()
      pose_stamped.pose.position.x = x[i]
      pose_stamped.pose.position.y = y[i]
      pub_msg.poses.append(pose_stamped)
      
    self.global_path_pub.publish(pub_msg)


def return_path(current_node):
  path = []
  current = current_node
  
  while current is not None:
    path.append(current.position)
    current = current.parent
    
  return path[::-1]  # return reversed path


def main():
  grid_size = 0.4
  robot_radius = 2.0
  # start_position = (10, 40)
  # goal_position = (90, -40)
  # start_position = (10, -15)
  # goal_position = (45, -15)
  start_position = (3, 0)
  goal_position = (45, 0)
  
  scene_num = 1
  plot_point = True
  
  show_animation = True

  ros = ROS()
  astar = Astar(grid_size, robot_radius, start_position, goal_position, show_animation, scene_num)
  
  # astar.erode_obstacles()
  # astar.dilate_obstacles()
  
  fig, ax = plt.subplots()
  ax.grid()
  ax.axis('equal')
  
  start_time = time.time()

  while astar.astar_condition():
    astar.get_the_current_node()
    path_node = astar.check_find_the_goal()
    astar.generate_children()
    astar.loop_through_children()

    if path_node is not None:
      end_time = time.time()
      print('\nterminate_time: {}s'.format(end_time - start_time))
      break
  
  if path_node is None:
    print('There is no Path')
    sys.exit()

  result = np.ones((len(path_node), 2))

  for i in range(len(path_node)):
    result[i][1] = ((len(astar.maze) / 2 * astar.grid_size) - path_node[i][0] * astar.grid_size)
    result[i][0] = path_node[i][1] * astar.grid_size

  rx, ry = [], []

  for i in range(len(result)):
    rx.append(result[i][0])
    ry.append(result[i][1])

  r = rx + ry

  path_len = len(rx)
  path = np.reshape(r, (2, path_len))

  print('Global Path')
  print('Start_node : {}'.format(start_position))
  print('Goal_node  : {}'.format(goal_position))
  print('path_length: {}'.format(path_len / 2))
  print('N of nodes : {}'.format(path_len))
  print('grid_size  : {}'.format(grid_size))
  print('map size   : {} x {}'.format(astar.x_width, astar.y_width))

  ox = []
  oy = []
  
  ox_grid = []
  oy_grid = []

  for i in range(len(astar.obstacles)):
    ox.append(astar.obstacles[i].x)
    oy.append(astar.obstacles[i].y)
    
  for i in range(len(astar.obs_grid)):
    ox_grid.append(astar.obs_grid[i].x)
    oy_grid.append(astar.obs_grid[i].y)

  if show_animation:
    plt.figure(figsize = (10, 3))
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.ylim(-10, 10)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.plot(rx, ry, '+c')
    plt.scatter(rx[0], ry[0], color = 'red', s = 150)
    plt.scatter(rx[-1], ry[-1], color = 'green', s = 150)
    if plot_point:
      plt.plot(ox, oy, ".k")
    plt.scatter(ox_grid, oy_grid, marker='s', color='gray', s = 400)
    # plt.grid(True)
    plt.axis('equal')
    plt.pause(0.001)
    plt.show()
    

  r = rospy.Rate(10)

  while not rospy.is_shutdown():
    ros.path_publish(path[0], path[1])
    r.sleep()
    
    

if __name__ == '__main__':
  main()