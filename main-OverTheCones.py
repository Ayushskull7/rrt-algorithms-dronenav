import enum
import numpy as np
import json
from rrt_algorithms.rrt.rrt_base import RRTBase
from rrt_algorithms.utilities.geometry import steer
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot
import os


class Status(enum.Enum):
    FAILED = 1
    TRAPPED = 2
    ADVANCED = 3
    REACHED = 4


class RRTConnect(RRTBase):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01):
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc)
        self.swapped = False

    def swap_trees(self):
        self.trees[0], self.trees[1] = self.trees[1], self.trees[0]
        self.swapped = not self.swapped

    def unswap(self):
        if self.swapped:
            self.swap_trees()

    def extend(self, tree, x_rand, enforce_waypoint=False, waypoint=None):
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = steer(x_nearest, x_rand, self.q)

        if enforce_waypoint and waypoint:
            if np.linalg.norm(np.array(x_new) - np.array(waypoint)) < self.q:
                x_new = waypoint

        if self.connect_to_point(tree, x_nearest, x_new):
            if np.abs(np.sum(np.array(x_new) - np.array(x_rand))) < 1e-2:
                return x_new, Status.REACHED
            return x_new, Status.ADVANCED
        return x_new, Status.TRAPPED

    def connect(self, tree, x):
        S = Status.ADVANCED
        while S == Status.ADVANCED:
            x_new, S = self.extend(tree, x)
        return x_new, S

    def rrt_connect(self, waypoint=None):
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)
        self.add_tree()
        self.add_vertex(1, self.x_goal)
        self.add_edge(1, self.x_goal, None)

        while self.samples_taken < self.max_samples:
            x_rand = self.X.sample_free()

            x_new, status = self.extend(0, x_rand, enforce_waypoint=True, waypoint=waypoint)
            if status != Status.TRAPPED:
                x_new, connect_status = self.connect(1, x_new)
                if connect_status == Status.REACHED:
                    self.unswap()
                    first_part = self.reconstruct_path(0, self.x_init, self.get_nearest(0, x_new))
                    second_part = self.reconstruct_path(1, self.x_goal, self.get_nearest(1, x_new))
                    second_part.reverse()
                    return first_part + second_part
            self.swap_trees()
            self.samples_taken += 1


repo_dir = os.path.dirname(os.path.abspath(__file__))

detections_file_path = os.path.join(repo_dir, "detections.json")

with open(detections_file_path, "r") as file:
    data = json.load(file)

detections = data[0]["detections"]
obstacles = [
    (bbox[0], bbox[1], bbox[2], bbox[3])
    for detection in detections
    if detection["class"] == "Obstacles" and detection["confidence"] > 0.5
    for bbox in detection["bbox"]
]
cones = [
    (bbox[0], bbox[1], bbox[2], bbox[3])
    for detection in detections
    if detection["class"] == "Cones" and detection["confidence"] > 0.5
    for bbox in detection["bbox"]
]

X_dimensions = np.array([(0, 1280), (0, 1280)])

all_obstacles = np.array(obstacles)

x_init = (1280, 0)
x_goal = (400, 1000)

cone1 = cones[0]
cone2 = cones[1]

cone1_center = ((cone1[0] + cone1[2]) / 2, (cone1[1] + cone1[3]) / 2)
cone2_center = ((cone2[0] + cone2[2]) / 2, (cone2[1] + cone2[3]) / 2)

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

distance_to_cone1 = euclidean_distance(x_init, cone1_center)
distance_to_cone2 = euclidean_distance(x_init, cone2_center)

if distance_to_cone1 < distance_to_cone2:
    waypoint_1 = cone1_center
    waypoint_2 = cone2_center
else:
    waypoint_1 = cone2_center
    waypoint_2 = cone1_center

q = 50
r = 5
max_samples = 2048
prc = 0.1

X = SearchSpace(X_dimensions, all_obstacles)

rrt_connect = RRTConnect(X, q, x_init, waypoint_1, max_samples, r, prc)
path_to_cone1 = rrt_connect.rrt_connect(waypoint=waypoint_1)

rrt_connect = RRTConnect(X, q, waypoint_1, waypoint_2, max_samples, r, prc)
path_to_cone2 = rrt_connect.rrt_connect(waypoint=waypoint_2)

rrt_connect = RRTConnect(X, q, waypoint_2, x_goal, max_samples, r, prc)
path_to_goal = rrt_connect.rrt_connect()

final_path = path_to_cone1 + path_to_cone2 + path_to_goal

plot = Plot("Drone Path Over Cones Avoiding Obstacles")
plot.plot_tree(X, rrt_connect.trees)
if final_path is not None:
    plot.plot_path(X, final_path)
plot.plot_obstacles(X, all_obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.plot_waypoint(X, waypoint_1)
plot.plot_waypoint(X, waypoint_2)

for cone in cones:
    cone_center = ((cone[0] + cone[2]) / 2, (cone[1] + cone[3]) / 2)
    plot.plot_cones(X, cone_center)

plot.draw(auto_open=True)
