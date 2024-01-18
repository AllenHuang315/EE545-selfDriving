import math
import numpy
from matplotlib import pyplot as plt
import cv2
import Utils
import time
import random
import waypoints
import FinalPlanPath
import FinalPlanPathNew
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped, Point

class HaltonPlanner(object):
  
  # planningEnv: Should be a HaltonEnvironment
  def __init__(self, planningEnv):
    self.planningEnv = planningEnv

    self.graph_pub = rospy.Publisher("/graph", MarkerArray, queue_size = 1)
    

  def show_waypoints(self):
    marker_array = MarkerArray()
    visted=set()
    id = 0
    
    for i in numpy.array(waypoints.good_waypoints, dtype=numpy.float64):
      config = Utils.map_to_world(i, self.planningEnv.manager.map_info)
      marker = Marker()
      marker.id = id
      marker.header.frame_id = "/map"
      marker.type = marker.SPHERE
      marker.action = marker.ADD
      marker.scale.x = 0.2
      marker.scale.y = 0.2
      marker.scale.z = 0.2
      marker.color.a = 1.0
      marker.color.r = 0.0
      marker.color.g = 0.0
      marker.color.b = 1.0
      marker.pose.orientation.w = 1.0
      marker.pose.position.x = config[0]
      marker.pose.position.y = config[1]
      marker.pose.position.z = 0
      marker_array.markers.append(marker)
      id += 1
    for i in numpy.array(waypoints.bad_waypoints, dtype=numpy.float64):
      config = Utils.map_to_world(i, self.planningEnv.manager.map_info)
      marker = Marker()
      marker.id = id
      marker.header.frame_id = "/map"
      marker.type = marker.SPHERE
      marker.action = marker.ADD
      marker.scale.x = 0.2
      marker.scale.y = 0.2
      marker.scale.z = 0.2
      marker.color.a = 1.0
      marker.color.r = 1.0
      marker.color.g = 0.0
      marker.color.b = 0.0
      marker.pose.orientation.w = 1.0
      marker.pose.position.x = config[0]
      marker.pose.position.y = config[1]
      marker.pose.position.z = 0
      marker_array.markers.append(marker)
      id += 1
    print('publish marker array..............')
    self.graph_pub.publish(marker_array)

  # Generate a plan
  # Assumes that the source and target were inserted just prior to calling this
  # Returns the generated plan
  def plan(self):
    self.show_waypoints()
    start_time = time.time()
    self.sid = self.planningEnv.graph.number_of_nodes() - 2 # Get source id
    self.tid = self.planningEnv.graph.number_of_nodes() - 1 # Get target id

    self.closed = {} # The closed list
    self.parent = {self.sid:None} # A dictionary mapping children to their parents
    self.open = {self.sid: 0 + self.planningEnv.get_heuristic(self.sid, self.tid)} # The open list
    self.gValues = {self.sid:0} # A mapping from node to shortest found path length to that node 
    self.planIndices = []
    self.cost = 0
    # ------------------------------------------------------------
    # YOUR CODE HERE
    # 
    # Implement A*
    # Functions that you will probably use
    # - self.get_solution()
    # - self.planningEnv.get_successors()
    # - self.planningEnv.get_distance()
    # - self.planningEnv.get_heuristic()
    # Note that each node in the graph has both an associated id and configuration
    # You should be searching over ids, not configurations. get_successors() will return
    #   the ids of nodes that can be reached. Once you have a path plan
    #   of node ids, get_solution() will compute the actual path in SE(2) based off of
    #   the node ids that you have found.
    #-------------------------------------------------------------
    while self.open:
      cur_node_id = min(self.open, key = self.open.get)
      if cur_node_id == self.tid:
        break
      self.closed[cur_node_id] = self.open.pop(cur_node_id)
      for successor_id in self.planningEnv.get_successors(cur_node_id):
        if successor_id not in self.closed:
          temp_g = self.gValues[cur_node_id] + self.planningEnv.get_distance(cur_node_id, successor_id)
          if successor_id not in self.gValues or self.gValues[successor_id] > temp_g:
            cur_node_config = self.planningEnv.get_config(cur_node_id)
            successor_config = self.planningEnv.get_config(successor_id)
            if self.planningEnv.manager.get_edge_validity(cur_node_config, successor_config):
              self.gValues[successor_id] = temp_g
              self.parent[successor_id] = cur_node_id
              self.open[successor_id] = temp_g + self.planningEnv.get_heuristic(successor_id, self.tid)
    end_time = time.time()
    path = self.get_solution(cur_node_id)
    print('The algorithm used {} seconds to plan the path, the path has {} nodes(s) and the cost is {}'.format(end_time-start_time, len(path), self.gValues[cur_node_id]))
    return path

  # Try to improve the current plan by repeatedly checking if there is a shorter path between random pairs of points in the path
  def post_process(self, plan, timeout):

    t1 = time.time()
    original_cost = self.cost
    elapsed = 0
    while elapsed < timeout: # Keep going until out of time
      # ---------------------------------------------------------
      # YOUR CODE HERE
      
      # Pseudocode
      
      # Pick random id i
      # Pick random id j
      # Redraw if i == j
      # Switch i and j if i > j
     
      # if we can find path between i and j (Hint: look inside ObstacleManager.py for a suitable function)
        # Get the path
        # Reformat the plan such that the new path is inserted and the old section of the path is removed between i and j
        # Be sure to CAREFULLY inspect the data formats of both the original plan and the plan returned
        # to ensure that you edit the path correctly
      i = numpy.random.randint(0, len(plan)-1)
      j = numpy.random.randint(0, len(plan)-1)
      if i == j:
        continue
      i, j = min(i,j), max(i, j)

      config_i, config_j = plan[i], plan[j]
      if self.planningEnv.manager.get_edge_validity(config_i, config_j):
        list_x, list_y, new_cost = self.planningEnv.manager.discretize_edge(config_i, config_j)
        cur_cost = sum(numpy.linalg.norm(numpy.array(plan[p+1]) - numpy.array(plan[p])) for p in range(i, j+1))
        if cur_cost > new_cost:
          plan = plan[:i] + ([list(points) for points in zip(list_x, list_y)]) + plan[j+1:]
      elapsed = time.time() - t1
    self.cost = sum(self.planningEnv.manager.discretize_edge(plan[p], plan[p+1])[2] for p in range(len(plan) - 1))
    print('cost after post process is: {}'.format(self.cost))
    print('original cost was: {}'.format(original_cost))
    # return FinalPlanPath.final_plan_path
    return FinalPlanPathNew.final_plan_path
    # return plan

  # Backtrack across parents in order to recover path
  # vid: The id of the last node in the graph
  def get_solution(self, vid):

    # Get all the node ids
    planID = []
    while vid is not None:
      planID.append(vid)
      vid = self.parent[vid]

    plan = []
    planID.reverse()
    for i in range(len(planID) - 1):
      startConfig = self.planningEnv.get_config(planID[i])
      goalConfig = self.planningEnv.get_config(planID[i + 1])
      px, py, clen = self.planningEnv.manager.discretize_edge(startConfig, goalConfig)
      plan.append([list(a) for a in zip(px, py)])
      self.planIndices.append(len(plan))
      self.cost += clen

    flatPlan = [item for sublist in plan for item in sublist]
    return flatPlan

  # Visualize the plan
  def simulate(self, plan):
    # Get the map
    envMap = 255*(self.planningEnv.manager.mapImageBW+1) # Hacky way to get correct coloring
    envMap = cv2.cvtColor(envMap, cv2.COLOR_GRAY2RGB)
    
    for i in range(numpy.shape(plan)[0]-1): # Draw lines between each configuration in the plan
      startPixel = Utils.world_to_map(plan[i], self.planningEnv.manager.map_info)
      goalPixel = Utils.world_to_map(plan[i+1], self.planningEnv.manager.map_info)
      cv2.line(envMap,(startPixel[0],startPixel[1]),(goalPixel[0],goalPixel[1]),(255,0,0),5)

    # Generate window
    cv2.namedWindow('Simulation', cv2.WINDOW_NORMAL)
    cv2.imshow('Simulation', envMap)

    # Terminate and exit elegantly
    cv2.waitKey(20000)
    cv2.destroyAllWindows()
