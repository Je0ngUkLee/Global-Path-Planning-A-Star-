#!/usr/bin/env python
# -*- coding: utf-8 -*-

from warnings import warn

import signal
import sys
import timeit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def signal_handler(signal, frame):
  print('\nYou pressed Ctrl+C!')
  sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def return_path(current_node):
  
  path = []
  current = current_node
  
  while current is not None:
    path.append(current.position)
    current = current.parent
    
  return path[::-1] # return reversed path


class Node():
  
  def __init__(self, parent = None, position = None):
    
    self.parent   = parent
    self.position = position
    self.G = 0; self.H = 0; self.T = 0; self.A = 0; self.F = 0
    
  def __eq__(self, other):
    return self.position == other.position
  
  
class Obstacle():
  def __init__(self, x, y, r):
    self.x = x; self.y = y; self.r = r
    self.pos = (0, 0)
    
    
class Astar():
  
  def __init__(self, grid_size):
    
    self.grid_size    = grid_size  #[m]
    self.robot_radius = 1  #[m]
    self.allow_diagonal_movement = True
    
    # map size
    self.x_width = 30  # (x: 0 ~ 9)
    self.y_width = 30
        
    self.maze = [[0 for col in range(int(self.x_width / self.grid_size) + 1)] for row in range(int(self.y_width / self.grid_size) + 1)]
    
    self.start_position = (0, 2)
    self.goal_position  = (20, 0)
    
    print('{} -> {}'.format(self.start_position, self.goal_position))
    
    self.start_node = Node(None, self.calc_grid_pos(self.start_position))
    self.goal_node  = Node(None, self.calc_grid_pos(self.goal_position))
    
    self.open_list   = []
    self.closed_list = []
    
    self.open_list.append(self.start_node)
    
    self.outer_iteration = 0
    self.max_iteration = (len(self.maze[len(self.maze) - 1]) // 2) ** 2
    
    self.adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0))
    if self.allow_diagonal_movement:
      self.adjacent_squares = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
      
    self.obstacles = [Obstacle(5, 2, 1), Obstacle(5, 1, 1), Obstacle(5, 0, 1), Obstacle(5, -1, 1), Obstacle(5, -2, 1)]  #Obstacle(x, y, r)
    self.update_obstacles()
    
    self.collision_node_1 = []
    self.collision_node_2 = []
    self.collision_node_3 = []
    
    
  def calc_grid_pos(self, node):
    # return ( ((len(self.maze) / 2) - node[1]) / self.grid_size , node[0] / self.grid_size )
    return ( ((len(self.maze) / 2 * self.grid_size) - node[1]) / self.grid_size , node[0] / self.grid_size )
  
  def update_obstacles(self):
    obstacle_adjacent_squares = ((1, 1), (1, -1), (-1, -1), (-1, 1))
    if self.obstacles is not None:
      for obs in self.obstacles:
        obs.pos = self.calc_grid_pos((obs.x, obs.y))
        
        if obs.r > self.grid_size:
          k     = (obs.r - self.grid_size) / self.grid_size
          point = []
          for oas in obstacle_adjacent_squares:
            point.append((oas[0] * k, oas[1] * k))
            
          ## 장애물에 맞춰 사각형 그리기 (장애물로 인한 그리드 점유)          
          minimum = int(point[1][1])
          maximum = int(point[0][0])
          
          for i in range(minimum, maximum + 1, 1):
            for j in range(minimum, maximum + 1, 1):
              if self.check_range((obs.pos[0] + i, obs.pos[1] + j)):
                continue
              self.maze[int(obs.pos[0]) + i][int(obs.pos[1]) + j] = 9
              
        if self.check_range(obs.pos):
          continue
        self.maze[int(obs.pos[0])][int(obs.pos[1])] = 9
            
        # while 1:
        #   for j in range(len(self.adjacent_squares)):
        #     if self.check_range((self.obstacles[i].pos[0] + k * self.adjacent_squares[j][0], self.obstacles[i].pos[1] + k * self.adjacent_squares[j][1])):
        #       continue
        #     self.maze[int(self.obstacles[i].pos[0] + k * self.adjacent_squares[j][0])][int(self.obstacles[i].pos[1] + k * self.adjacent_squares[j][1])] = 9
        #   cnt += self.grid_size
        #   k += 1
        #   if cnt >= self.obstacles[i].r:
        #     break
          
  def check_range(self, node):
    self.within_range_criteria = [
      node[0] > (len(self.maze) - 1),
      node[0] < 0,
      node[1] > (len(self.maze[len(self.maze) - 1]) - 1),
      node[1] < 0
    ]
    if any(self.within_range_criteria):
      return True
    
    
  def astar_condition(self):
    return len(self.open_list) > 0
  
  def get_the_current_node(self):
    self.current_node  = self.open_list[0]
    self.current_index = 0
    
    for index, item in enumerate(self.open_list):
      if item.F < self.current_node.F:
        self.current_node  = item
        self.current_index = index
        
      elif item.F == self.current_node.F:
        if item.H < self.current_node.H:
          self.current_node  = item
          self.current_index = index
          
    if self.outer_iteration > self.max_iteration:
      # If we hit this point return the path such as it is
      # It will not contain the destination
      warn("giving up on pathfinding too many iterations")
      return return_path(self.current_node)
      # return self.return_path()
      
    self.open_list.pop(self.current_index)
    self.closed_list.append(self.current_node)
    
  def check_find_the_goal(self):
    if self.current_node == self.goal_node:
      return return_path(self.current_node)
    
  def generate_children(self):
    self.children = []
    
    for new_position in self.adjacent_squares:
      self.node_position = (self.current_node.position[0] + new_position[0], self.current_node.position[1] + new_position[1])
      
      if self.check_range(self.node_position):
        continue
      if self.maze[int(self.node_position[0])][int(self.node_position[1])] != 0:
        continue
      
      self.new_node = Node(self.current_node, self.node_position)
      self.children.append(self.new_node)
      
  def loop_through_children(self):
    
    for child in self.children:
      
      # Child is already in the closed list
      if len([closed_child for closed_child in self.closed_list if closed_child == child]) > 0:
        continue
      
      # Create the G, H, F values
      if (child.position[0] == self.current_node.position[0] or child.position[1] == self.current_node.position[1]):
        child.G = self.current_node.G + 10
      else:
        child.G = self.current_node.G + 14
      child.H = (abs(child.position[0] - self.goal_node.position[0]) + abs(child.position[1] - self.goal_node.position[1]) ) * 10
      child.F = child.G + child.H
      
      # Child is already in the open list
      if len([open_node for open_node in self.open_list if child == open_node and child.G > open_node.G]) > 0:
        continue
      
      # Add the child to the open list
      self.open_list.append(child)


def main():
  print(__file__ + " Start !!")
  
  astar = Astar(grid_size = 0.5)  # grid_size: 0.25, 0.5, 1.0, ...
  show_animation = True
  start_time = timeit.default_timer()
  
  while astar.astar_condition():    
    
    astar.get_the_current_node()
    path_node = astar.check_find_the_goal()
    astar.generate_children()
    astar.loop_through_children()
    
    if path_node is not None:
      end_time = timeit.default_timer()
      print('terminate_time: {}s'.format(end_time - start_time))
      break
    
  for i in range(len(astar.maze)):
    print(astar.maze[i])
    
  result = np.ones((len(path_node), 2))
  
  # ## result: 최종 경로, (x, y)형태
  
  for i in range(len(path_node)):
    result[i][1] = (len(astar.maze) / 2 * astar.grid_size) - path_node[i][0] * astar.grid_size
    result[i][0] = path_node[i][1] * astar.grid_size
    
  # print('\n')
  # print(result.T)
    
  rx, ry = [], []
  
  for i in range(len(result)):
    rx.append(result[i][0])
    ry.append(result[i][1])
    
  r = rx + ry
    
  path_len = len(rx)
  path = np.reshape(r, (2, path_len))

  print('Global Path')
  print('x: {}\ny: {}'.format(path[0], path[1]))
  
  # if show_animation:
  #   plt.plot(rx, ry, "-r")
  #   plt.pause(2.0)
  #   plt.show()
    
  
  
if __name__ == '__main__':
  main()