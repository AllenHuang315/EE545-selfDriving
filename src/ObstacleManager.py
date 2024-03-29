import cv2
import math
import numpy
import Utils

class ObstacleManager(object):

	def __init__(self, mapMsg, car_width, car_length, collision_delta):
		self.map_info = mapMsg.info
		self.mapImageGS = numpy.array(mapMsg.data, dtype=numpy.uint8).reshape(
			(mapMsg.info.height, mapMsg.info.width, 1))

		# Retrieve the map dimensions
		height, width, channels = self.mapImageGS.shape
		self.mapHeight = height
		self.mapWidth = width
		self.mapChannels = channels

		# Binarize the Image
		self.mapImageBW = 255 * numpy.ones_like(self.mapImageGS, dtype=numpy.uint8)
		self.mapImageBW[self.mapImageGS == 0] = 0

		# Obtain the car length and width in pixels
		self.robotWidth = int(car_width / self.map_info.resolution + 0.5)
		self.robotLength = int(car_length / self.map_info.resolution + 0.5)
		self.collision_delta = collision_delta

	# Check if the passed config is in collision
	# config: The configuration to check (in meters and radians)
	# Returns False if in collision, True if not in collision
	def get_state_validity(self, config):

		# Convert the configuration to map-coordinates -> mapConfig is in pixel-space
		mapConfig = Utils.world_to_map(config, self.map_info)

		# ---------------------------------------------------------
		# YOUR CODE HERE
		#
		# Return true or false based on whether the robot's configuration is in collision
		# Use a square to represent the robot, return true only when all points within
		# the square are collision free
		#
		# Also return false if the robot is out of bounds of the map
		#
		# Although our configuration includes rotation, assume that the
		# square representing the robot is always aligned with the coordinate axes of the
		# map for simplicity
		# ----------------------------------------------------------
		half_diag = numpy.sqrt(self.robotWidth ** 2 + self.robotLength ** 2) / 2
		left, right = int(numpy.ceil(mapConfig[0] - half_diag)), int(numpy.ceil(mapConfig[0] + half_diag))
		rear, front = int(numpy.ceil(mapConfig[1] - half_diag)), int(numpy.ceil(mapConfig[1] + half_diag))

		if left <= 0 or right <= 0 or rear <= 0 or front <= 0:
			return False
		if left >= self.mapWidth or right >= self.mapWidth or front >= self.mapHeight or rear >= self.mapHeight:
			return False
		return not numpy.any(self.mapImageBW[rear:front, left:right])

	# Discretize the path into N configurations, where N = path_length / self.collision_delta
	#
	# input: an edge represented by the start and end configurations
	#
	# return three variables:
	# list_x - a list of x values of all intermediate points in the path
	# list_y - a list of y values of all intermediate points in the path
	# edgeLength - The euclidean distance between config1 and config2
	def discretize_edge(self, config1, config2):
		list_x, list_y = [], []
		edgeLength = 0
		# -----------------------------------------------------------
		# YOUR CODE HERE
		edge_vector = numpy.array(config1) - numpy.array(config2)
		edgeLength = numpy.linalg.norm(edge_vector)
		N = int(numpy.ceil(edgeLength / self.collision_delta))
		list_x = numpy.linspace(config1[0], config2[0], N)
		list_y = numpy.linspace(config1[1], config2[1], N)
		# -----------------------------------------------------------
		return list_x, list_y, edgeLength


	# Check if there is an unobstructed edge between the passed configs
	# config1, config2: The configurations to check (in meters and radians)
	# Returns false if obstructed edge, True otherwise
	def get_edge_validity(self, config1, config2):
		# -----------------------------------------------------------
		# YOUR CODE HERE
		#
		# Check if endpoints are obstructed, if either is, return false
		# Find path between two configs by connecting them with a straight line
		# Discretize the path with the discretized_edge function above
		# Check if all configurations along path are obstructed
		# -----------------------------------------------------------
		if not self.get_state_validity(config1) or not self.get_state_validity(config2):
			return False
		list_x, list_y, _ = self.discretize_edge(config1, config2)

		return all(self.get_state_validity(location) for location in zip(list_x, list_y))


# Write Your Test Code For Debugging
#if __name__ == '__main__':
#	return
	# Write test code here!
