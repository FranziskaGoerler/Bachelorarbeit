import arcade
import gym
# import random
import timeit
# import math
import numpy as np
import time
import bisect
from matplotlib import pyplot as plt
from multiple_formatter import multiple_formatter

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
N_BINS = 4
N_BOTS = 6
BOT_RADIUS = 17.5
COLLISION_DISTANCE = BOT_RADIUS + 17.5
PLAN_STEPS = 20         # default 45
CHECK_MAX_STEPS = 5    # default 15
RADIUS = 100
MIN_DIST = 20
SCREEN_TITLE = "pycking environment"

ROBOT_MAX_SPEED = 0    # if exactly 0, step function of robot is never called, which speeds up environment
AGENT_MAX_SPEED = 8
PUNISH_WRONG_DIRECTION = True
WRONG_DIRECTION_CONSTANT = - 0.5   # additional constant punishment. Wrong direction punishment is a linear function, this is added.
REWARD_TARGET_FOUND = 50    # reward for reaching target
REWARD_COLLISION = -100
REWARD_BOUNDARY = -100
TARGET_GAUSS_STD = np.pi/4
ROBOT_GAUSS_STD = np.pi/12
ROBOT_DISTANCE_SCALING_CONSTANT = 60    # distance scaling is -(2 / np.exp(800/C)) * np.exp((800-x)/C)
ROBOT_REWARD_MAX = -4
N_SENSORS = 36
DONE_AT_COLLISION = True
IGNORE_ROBOTS = False
AGENT_MAX_STEPS = 350    # max length of an episode
TARGET_INDEX = None      # Index of target for agent. None for random target each episode.
CENTER_START = False   # Whether to always put agent in the screen center (True) or initialize randomly (False)

# DESCRIPTION = ''
DESCRIPTION = 'observation: agent pos, target vector, 36 gauss sensors. Learnrate decreased by 10.' \
              'Sensors are exactly the part of the reward function that is caused by robots.'

param_names = ['SCREEN_WIDTH', 'SCREEN_HEIGHT', 'N_BINS', 'N_BOTS', 'BOT_RADIUS', 'COLLISION_DISTANCE',
               'PLAN_STEPS', 'CHECK_MAX_STEPS', 'RADIUS', 'MIN_DIST', 'SCREEN_TITLE', 'ROBOT_MAX_SPEED',
               'AGENT_MAX_SPEED', 'PUNISH_WRONG_DIRECTION', 'WRONG_DIRECTION_CONSTANT', 'REWARD_TARGET_FOUND',
               'REWARD_COLLISION', 'REWARD_BOUNDARY', 'TARGET_GAUSS_STD', 'ROBOT_GAUSS_STD', 'ROBOT_DISTANCE_SCALING_CONSTANT', 'ROBOT_REWARD_MAX',
               'N_SENSORS', 'DONE_AT_COLLISION', 'IGNORE_ROBOTS', 'AGENT_MAX_STEPS', 'TARGET_INDEX', 'CENTER_START', 'DESCRIPTION']

env_params = {p: eval(p) for p in param_names}

class Shape:
    """ Generic base shape class """
    def __init__(self, x, y, width, height, angle, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
        self.color = color
        self.shape_list = None

    def draw(self):
        pass


class BufferedShape(Shape):
    def __init__(self, x, y, width, height, angle, color):
        super().__init__(x, y, width, height, angle, color)

    def draw(self):
        self.shape_list.center_x = self.x
        self.shape_list.center_y = self.y
        self.shape_list.angle = self.angle
        self.shape_list.draw()


class Line(Shape):
    def __init__(self, point_list=()):
        self._line_width = 2
        super().__init__(point_list, list(range(len(point_list))), self._line_width, 0, 0, arcade.color.BLACK)

    def draw(self):
        # drawing the lines is unbuffered because each line segment may change between frames
        # it might make sense to have an option to disable line drawing completely for performance boost
        arcade.draw_line_strip(self.x, self.color, self.width)


class RobotShape(BufferedShape):
    def __init__(self, xpos, ypos):
        self._radius = BOT_RADIUS
        super().__init__(xpos, ypos, self._radius, self._radius, 0, arcade.color.GREEN)
        shape = arcade.create_ellipse_filled(0, 0,
                                             self.width, self.height,
                                             self.color, self.angle)
        self.shape_list = arcade.ShapeElementList()
        self.shape_list.append(shape)

        point_list = ((self.x, self.y), (self.x + 10, self.y + 10), (self.x + 14, self.y + 18))
        self.planned_path = Line(point_list)

    def draw(self):
        super().draw()
        self.planned_path.draw()


def points_on_circumference(center=(0, 0), r=50, n=100):
    return [
        (
            center[0] + (np.cos(2 * np.pi / n * x) * r),  # x
            center[1] + (np.sin(2 * np.pi / n * x) * r)  # y

        ) for x in range(n)]

def gauss_func(x, mean=0, std=1):
    return np.exp(-((x-mean)/std)**2 / 2) / (std * np.sqrt(2*np.pi))

def x_wrap_around(x):
    x[x > np.pi] -= 2 * np.pi
    x[x < -np.pi] += 2 * np.pi
    return x

class Robot:
    def __init__(self, xpos, ypos, start_target, end_target):
        self.x = xpos
        self.y = ypos
        self.time = 0
        self.delta_average = 0
        self.wait = 0
        self.has_package = False
        self.pick_target = start_target
        self.drop_target = end_target
        # self.trajectory = points_on_circumference((400,400), 100, 100)
        # self.traj_index = 0

        point_list = ((self.x, self.y), (self.x + 10, self.y + 10), (self.x + 14, self.y + 18))
        # point_list = points_on_circumference((self.x, self.y), UNSAFE_DISTANCE, 100)
        # point_list = self.trajectory
        self.planned_path = Line(point_list)

    def step(self, dt, robots):
        # new_pos = self.trajectory[self.traj_index]
        # self.traj_index = (self.traj_index + 1) % len(self.trajectory)
        # self.x, self.y = new_pos[0], new_pos[1]
        # return
        if self.delta_average == 0:
            self.delta_average = dt
        else:
            self.delta_average = self.delta_average * 0.8 + dt * 0.2

        self.time += dt

        if self.wait > 0.0:
            self.wait -= dt
            if self.wait <= 0.0:
                if self.has_package:
                    # drop package and release target
                    self.drop_target.blocked = False
                    self.has_package = False
                    self.drop_target = None
                else:
                    # pick up package and release target
                    self.pick_target.blocked = False
                    self.has_package = True
                    self.pick_target = None
                self.wait = 0.0
            else:
                return

        self.plan(robots)

        if len(self.planned_path.x) > 1:
            self.x = self.planned_path.x[1][0]
            self.y = self.planned_path.x[1][1]

        target = self.drop_target if self.has_package else self.pick_target
        diffx, diffy = target.x - self.x, target.y - self.y

        if abs(diffx) <= MIN_DIST and abs(diffy) <= MIN_DIST:
            self.planned_path.x = [(self.x, self.y)]
            self.planned_path.y = [self.time]
            self.wait = 1.0

    def plan(self, robots):
        posx = self.x
        posy = self.y
        target = self.drop_target if self.has_package else self.pick_target

        t = self.time
        mv_plan = [(posx, posy)]
        mv_times = [t]

        for s in range(PLAN_STEPS):
            left_steps = PLAN_STEPS - s
            flag = False
            diffx, diffy = target.x - posx, target.y - posy
            angle = np.arctan2(diffy, diffx)
            d = 0
            max = np.pi * 2.0 / (s * 0.2 + 1)

            while d < max:
                sp = ROBOT_MAX_SPEED
                if np.sqrt(diffx**2+diffy**2) < RADIUS:
                    sp = ROBOT_MAX_SPEED / 2
                    if not target.blocked:
                        target.blocked = True
                        target.blocked_by = self
                    elif target.blocked_by != self:
                        sp = 0.0

                time = t + (s + 1) * self.delta_average

                vx = np.cos(angle + d) * sp
                vy = np.sin(angle + d) * sp
                if self.check_plan(robots, left_steps, posx, posy, vx, vy, time):
                    mv_plan.append((posx+vx, posy+vy))
                    mv_times.append(time)
                    posx += vx
                    posy += vy
                    dx, dy = target.x - posx, target.y - posy

                    if abs(dx) <= MIN_DIST and abs(dy) <= MIN_DIST:
                        self.planned_path.x = mv_plan
                        self.planned_path.y = mv_times
                        return

                    flag = True
                    break

                vx = np.cos(angle - d) * sp
                vy = np.sin(angle - d) * sp
                if self.check_plan(robots, left_steps, posx, posy, vx, vy, time):
                    mv_plan.append((posx + vx, posy + vy))
                    mv_times.append(time)
                    posx += vx
                    posy += vy
                    dx, dy = target.x - posx, target.y - posy

                    if abs(dx) <= MIN_DIST and abs(dy) <= MIN_DIST:
                        self.planned_path.x = mv_plan
                        self.planned_path.y = mv_times
                        return

                    flag = True
                    break

                d += 0.3

            if not flag:
                self.planned_path.x = mv_plan
                self.planned_path.y = mv_times
                return

            self.planned_path.x = mv_plan
            self.planned_path.y = mv_times

    def check_plan(self, robots, left_steps, posx, posy, vx, vy, time):
        steps = min(left_steps, CHECK_MAX_STEPS)

        for s in range(steps):
            pt = time + (s + 1) * self.delta_average
            px = posx + (s + 1) * vx
            py = posy + (s + 1) * vy

            if not self.validate(robots, px, py, pt):
                return False

        return True

    def validate(self, robots, posx, posy, time):
        for rob in robots:
            if rob != self:
                traj = rob.planned_path.x
                times = rob.planned_path.y
                for j in range(len(traj)):
                    if abs(times[j] - time) <= 1.5 * self.delta_average:
                        d = np.sqrt((traj[j][0] - posx)**2 + (traj[j][1] - posy)**2)
                        if d < COLLISION_DISTANCE:
                            return False
                last_index = len(traj) - 1
                if times[last_index] <= time:
                    d = np.sqrt((traj[last_index][0] - posx)**2 + (traj[last_index][1] - posy)**2)

                    if d < COLLISION_DISTANCE:
                        return False
        return True


class AgentShape(BufferedShape):
    def __init__(self, xpos, ypos):
        self._radius = 17.5
        super().__init__(xpos, ypos, self._radius, self._radius, 0, arcade.color.BLUE)
        shape = arcade.create_ellipse_filled(0, 0,
                                             self.width, self.height,
                                             self.color, self.angle)
        lin = arcade.create_line(0, 0, 0+self._radius, 0, arcade.color.WHITE, 5)
        self.shape_list = arcade.ShapeElementList()
        self.shape_list.append(shape)
        self.shape_list.append(lin)


class Agent:
    OK = 0
    TARGET_FOUND = 1
    BOUNDARY_COLLISION = 2
    ROBOT_COLLISION = 3

    def __init__(self, xpos, ypos, first_target, robots):
        self.x = xpos
        self.y = ypos
        # self.angle_pi6 = 0   # angle in multiples of pi/6, to avoid accumulating round-off error
        self.angle = 0
        self.angle_error = 0
        self.has_package = False
        self.target = first_target
        diffy, diffx = self.target.y - self.y, self.target.x - self.x
        self.angle_to_destination = np.arctan2(diffy, diffx)
        self.angle_error = self.angle_to_destination - self.angle
        self.dist_to_target = np.sqrt(diffy**2 + diffx**2)
        self._radius = 17.5
        # pi6 = np.pi/6
        # self.sensor_boundaries = [-2*pi6, -pi6, 0, pi6, 2*pi6, 3*pi6]
        # self.sensor_boundaries_all = [-5*pi6, -4*pi6, -3*pi6, -2*pi6, -pi6, 0, pi6, 2*pi6, 3*pi6, 4*pi6, 5*pi6, np.pi]
        self.gauss_sensor_x = np.arange(1, N_SENSORS+1) * (2*np.pi / N_SENSORS) - np.pi
        self.sensor_values = [0]*N_SENSORS
        # self.dist_to_rob = SCREEN_WIDTH+SCREEN_HEIGHT
        # self.dist_to_wall = min((self.x, SCREEN_WIDTH-self.x, self.y, SCREEN_HEIGHT-self.y))
        # self.update_sensors(robots)

        self.target_gauss_scaling = 1 / gauss_func(self.angle_to_destination, mean=self.angle_to_destination, std=TARGET_GAUSS_STD)

        self.robot_angles = []
        self.robot_distances = []
        self.robot_gauss = []
        self.robot_gauss_scaling = []
        for r in robots:
            C = ROBOT_DISTANCE_SCALING_CONSTANT
            diffx, diffy = r.x-self.x, r.y-self.y
            rob_angle = np.arctan2(diffy, diffx)
            rob_distance = np.sqrt(np.sum(diffx**2 + diffy**2))
            self.robot_angles.append(rob_angle)
            self.robot_distances.append(rob_distance)
            distance_to_collision = COLLISION_DISTANCE - rob_distance
            distance_scaling = -(2 / np.exp(800 / C)) * np.exp((800 - distance_to_collision) / C)  # heuristic
            rob_gauss_scaling = distance_scaling / gauss_func(rob_angle, mean=rob_angle, std=ROBOT_GAUSS_STD)
            self.robot_gauss_scaling.append(rob_gauss_scaling)

        self.update_gauss_sensors()

    def step(self, action, robots):

        # def distance_to_closest_robot(x, y, robots):
        #     d = SCREEN_HEIGHT+SCREEN_WIDTH
        #     d_min = SCREEN_HEIGHT+SCREEN_WIDTH
        #     for rob in robots:
        #         dx = rob.x - x
        #         dy = rob.y - y
        #         d = np.sqrt(dx**2 + dy**2)
        #
        #         if d <= COLLISION_DISTANCE:
        #             return 0
        #
        #         if d < d_min:
        #             d_min = d
        #
        #     return d_min

        action_length = np.sqrt(np.sum(action ** 2))
        if action_length > AGENT_MAX_SPEED:
            action = (action / action_length) * AGENT_MAX_SPEED
            action_length = AGENT_MAX_SPEED

        nx = self.x + action[0]
        ny = self.y + action[1]

        pre_diffx, pre_diffy = self.target.x - self.x, self.target.y - self.y

        if not IGNORE_ROBOTS:
            for ri, r in enumerate(robots):
                rob_diffx, rob_diffy = r.x - self.x, r.y - self.y
                self.robot_angles[ri] = np.arctan2(rob_diffy, rob_diffx)
                self.robot_distances[ri] = np.sqrt(np.sum(rob_diffx ** 2 + rob_diffy ** 2))
            min_dist_to_rob = min(self.robot_distances)
            if min_dist_to_rob <= COLLISION_DISTANCE:
                return Agent.ROBOT_COLLISION, REWARD_COLLISION
        min_dist_to_wall = min((nx, SCREEN_WIDTH-nx, ny, SCREEN_HEIGHT-ny))
        if min_dist_to_wall < 0:
            return Agent.BOUNDARY_COLLISION, REWARD_BOUNDARY

        self.x = nx
        self.y = ny

        diffx, diffy = self.target.x - self.x, self.target.y - self.y
        self.dist_to_target = np.sqrt(diffy**2 + diffx**2)

        if abs(diffx) <= MIN_DIST and abs(diffy) <= MIN_DIST:
            # target reached
            self.has_package = not self.has_package
            self.target.active = False
            self.target = None
            return Agent.TARGET_FOUND, REWARD_TARGET_FOUND

        # self.update_sensors(robots)

        self.angle_to_destination = np.arctan2(pre_diffy, pre_diffx)  # Winkel zum Ziel (von Ausgangsposition)
        action_angle = np.arctan2(action[1], action[0])
        self.angle_error = self.angle_to_destination - action_angle

        self.target_gauss_scaling = 1 / gauss_func(self.angle_to_destination, mean=self.angle_to_destination, std=TARGET_GAUSS_STD)
        reward = self.target_gauss_scaling * \
                                  gauss_func(action_angle, mean=self.angle_to_destination, std=TARGET_GAUSS_STD)
        if not IGNORE_ROBOTS:
            C = ROBOT_DISTANCE_SCALING_CONSTANT
            for ri, r in enumerate(robots):
                rob_angle = self.robot_angles[ri]
                distance_to_collision = self.robot_distances[ri] - COLLISION_DISTANCE
                distance_scaling = (ROBOT_REWARD_MAX / np.exp(800/C)) * np.exp((800 - distance_to_collision)/C)   # heuristic
                rob_gauss_scaling = distance_scaling/gauss_func(rob_angle, mean=rob_angle, std=ROBOT_GAUSS_STD)
                robot_reward = rob_gauss_scaling*gauss_func(action_angle, mean=rob_angle, std=ROBOT_GAUSS_STD)
                reward += robot_reward
                self.robot_gauss_scaling[ri] = rob_gauss_scaling

            if not IGNORE_ROBOTS:
                self.update_gauss_sensors()

        reward *= action_length

        if abs(self.angle_error) > np.pi/2 and PUNISH_WRONG_DIRECTION:
            reward = - action_length * ((2 / np.pi) * abs(self.angle_error) - (1 + WRONG_DIRECTION_CONSTANT))

        return Agent.OK, reward

    def update_gauss_sensors(self):
        robots_y = np.zeros(N_SENSORS)
        for ri in range(len(self.robot_angles)):
            rob_gauss_scaling = self.robot_gauss_scaling[ri]
            rob_angle = self.robot_angles[ri]
            rob_y = rob_gauss_scaling * gauss_func(self.gauss_sensor_x + rob_angle, mean=rob_angle, std=ROBOT_GAUSS_STD)
            rob_x = x_wrap_around(self.gauss_sensor_x + self.robot_angles[ri])
            rob_sortind = np.argsort(rob_x)
            rob_y = rob_y[rob_sortind]
            robots_y += rob_y
        self.sensor_values = robots_y

    # def update_sensors(self, robots):
    #     agent_angle = self.angle
    #     # if agent_angle > np.pi:
    #     #     agent_angle -= 2*np.pi   # angle should be in {-pi/2, pi/2}
    #     pi6 = np.pi/6
    #     angle_by_pi6 = agent_angle / pi6
    #     rotation_steps = int(np.ceil(angle_by_pi6))   # by how many steps entries in sensor list need to be shifted
    #     angle_delta = (angle_by_pi6 - rotation_steps) * pi6   # to subtract from sensor boundaries for calculation
    #     boundary_values = [0] * 12
    #     for ai, ang in enumerate(self.sensor_boundaries):
    #         sensor_angle = angle_delta + ang
    #         sin = np.sin(sensor_angle)
    #         cos = np.cos(sensor_angle)
    #         if abs(cos) > 1e-5:
    #             m = sin / cos  # y = mx + b  <->   b = y - mx
    #             b = self.y - m * self.x  # <-> x = (y-b)/m
    #             x1 = SCREEN_WIDTH
    #             y1 = m * x1 + b
    #             if y1 > SCREEN_HEIGHT:
    #                 y1 = SCREEN_HEIGHT
    #                 x1 = (y1 - b) / m
    #             value_right = np.sqrt((x1 - self.x) ** 2 + (y1 - self.y) ** 2)
    #             x0 = 0
    #             y0 = m * x0 + b
    #             if y0 < 0:
    #                 y0 = 0
    #                 x0 = (y0 - b) / m
    #             value_left = np.sqrt((x0 - self.x) ** 2 + (y0 - self.y) ** 2)
    #         else:
    #             # denominator almost 0 (vertical line) -> do sth else
    #             # x constant at self.x
    #             # y every value, including 0 and SCREEN_HEIGHT
    #             if ai > 0:
    #                 value_right = SCREEN_HEIGHT - self.y
    #                 value_left = self.y
    #             else:
    #                 value_left = SCREEN_HEIGHT - self.y
    #                 value_right = self.y
    #         boundary_values[(ai-rotation_steps) % 12] = value_right
    #         boundary_values[(ai+6-rotation_steps) % 12] = value_left
    #         boundary_values.append(boundary_values[0])  # tiny wrap-around
    #     for si in range(len(self.sensor_values)):
    #         self.sensor_values[si] = np.min(boundary_values[si:si + 2]) - 17.5
    #
    #     for r in robots:
    #         diffx, diffy = r.x - self.x, r.y - self.y
    #         angle_to_robot = np.arctan2(diffy, diffx)
    #         angle_r = angle_to_robot - angle_delta
    #         if angle_r > np.pi:
    #             angle_r -= 2*np.pi
    #         # distance is from center to center, but sensor would measure from edge to edge, so radii are subtracted
    #         distance_to_robot = np.sqrt(diffx**2 + diffy**2) - BOT_RADIUS - 17.5
    #         # sensor boundaries begin with -5*pi6. If bisect returns 0 that means angle < -5*pi6,
    #         # which is sensor number 8. So we add 8 to the index, and take modulo 12 to wrap indices around.
    #         sensor_index = (bisect.bisect(self.sensor_boundaries_all, angle_r) + 8) % 12
    #         self.sensor_values[sensor_index] = distance_to_robot


class TargetShape(BufferedShape):
    def __init__(self, xpos, ypos):
        self._width = 35
        super().__init__(xpos, ypos, self._width, self._width, 0, arcade.color.MAGENTA)
        shape = arcade.create_rectangle_filled(0, 0,
                                               self.width, self.height,
                                               self.color, self.angle)
        self.shape_list = arcade.ShapeElementList()
        self.shape_list.append(shape)

    def change_color(self, new_color):
        self.color = new_color
        shape = arcade.create_rectangle_filled(0, 0,
                                               self.width, self.height,
                                               self.color, self.angle)
        self.shape_list = arcade.ShapeElementList()
        self.shape_list.append(shape)


class Target:
    def __init__(self, xpos, ypos):
        self.x = xpos
        self.y = ypos
        self.blocked = False
        self.blocked_by = None
        self.active = False


class Wind(arcade.Window):
    def __init__(self, master, render=False, plot_reward=False):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.background_color = arcade.color.WHITE

        self.master = master

        self.draw_time = 0
        self.frame_count = 0
        self.fps_start_timer = None
        self.fps = None

        self.step_time = timeit.default_timer()

        self.robots = []
        self.start_targets = []
        self.end_targets = []
        self.agent = AgentShape(self.master.agent.x, self.master.agent.y)
        for r in self.master.robots:
            self.robots.append(RobotShape(r.x, r.y))
            self.robots[-1].planned_path = r.planned_path
        for t in self.master.start_targets:
            self.start_targets.append(TargetShape(t.x, t.y))
            if t.active:
                self.start_targets[-1].change_color(arcade.color.RED_ORANGE)
        for t in self.master.end_targets:
            self.end_targets.append(TargetShape(t.x, t.y))
            if t.active:
                self.end_targets[-1].change_color(arcade.color.RED_ORANGE)
        if plot_reward:
            pi = np.pi
            self.gauss_x = np.arange(1, 101) * (pi/50) - pi
            self.draw_x = np.array(self.gauss_x)
            # self.gauss_x_inner = self.gauss_x[24:75]
            # self.gauss_x_left = self.gauss_x[:24]
            # self.gauss_x_right = self.gauss_x[75:]

            self.target_y = np.zeros(100)
            self.robots_y = np.zeros(100)
            self.overall_y = np.zeros(100)

            plt.axis([-pi, pi, -2, 1])
            self.ax = plt.gca()
            self.ax.axhline(0, color='black', lw=1)
            self.ax.axvline(0, color='black', lw=1)
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
            self.ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
            self.ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
            plt.ion()
            plt.show()

            self.line_target, = self.ax.plot(self.draw_x, self.target_y, color='green')
            self.line_robot, = self.ax.plot(self.draw_x, self.robots_y, color='red')
            self.line_overall, = self.ax.plot(self.draw_x, self.overall_y, color='black')

        self.rendering = render
        self.plot_reward = plot_reward
        self.closed = False

        @self.event
        def on_close():
            # print("I'm closing now")
            self.closed = True
            self.close()

    def update_diagram_data(self):

        self.target_y = self.master.agent.target_gauss_scaling * gauss_func(
                        self.gauss_x + self.master.agent.angle_to_destination,
                        mean=self.master.agent.angle_to_destination, std=TARGET_GAUSS_STD)
        if PUNISH_WRONG_DIRECTION:
            self.target_y[:25] = - ((2 / np.pi) * abs(self.gauss_x[:25]) - (1 + WRONG_DIRECTION_CONSTANT))
            self.target_y[75:] = - ((2 / np.pi) * abs(self.gauss_x[75:]) - (1 + WRONG_DIRECTION_CONSTANT))

        self.robots_y = np.zeros(100)
        if not IGNORE_ROBOTS:
            for ri in range(len(self.master.robots)):
                # rob_gauss = self.master.agent.robot_gauss[ri]
                rob_gauss_scaling = self.master.agent.robot_gauss_scaling[ri]
                rob_angle = self.master.agent.robot_angles[ri]
                rob_y = rob_gauss_scaling * gauss_func(self.gauss_x + rob_angle, mean=rob_angle, std=ROBOT_GAUSS_STD)
                rob_x = x_wrap_around(self.gauss_x + self.master.agent.robot_angles[ri])
                rob_sortind = np.argsort(rob_x)
                rob_y = rob_y[rob_sortind]
                self.robots_y += rob_y

        self.draw_x = x_wrap_around(self.gauss_x + self.master.agent.angle_to_destination)
        sortind = np.argsort(self.draw_x)
        self.draw_x = self.draw_x[sortind]
        self.target_y = self.target_y[sortind]
        self.overall_y = self.target_y + self.robots_y

    def on_update(self, dt):

        def color_check(t, tS):
            if t.active and tS.color == arcade.color.MAGENTA:
                tS.change_color(arcade.color.RED_ORANGE)
            elif not t.active and tS.color == arcade.color.RED_ORANGE:
                tS.change_color(arcade.color.MAGENTA)

        self.agent.x = self.master.agent.x
        self.agent.y = self.master.agent.y
        # angle is in rad, but arcade used deg angle. This has to be converted here
        self.agent.angle = self.master.agent.angle*(180 / np.pi)
        for r, rS in zip(self.master.robots, self.robots):
            rS.x = r.x
            rS.y = r.y
            rS.planned_path = r.planned_path
        for t, tS in zip(self.master.start_targets, self.start_targets):
            tS.x = t.x
            tS.y = t.y
            color_check(t, tS)
        for t, tS in zip(self.master.end_targets, self.end_targets):
            tS.x = t.x
            tS.y = t.y
            color_check(t, tS)

        if self.plot_reward:
            self.update_diagram_data()

    def on_draw(self):
        """
        Render the screen.
        """
        # Start timing how long this takes
        draw_start_time = timeit.default_timer()

        if self.frame_count % 60 == 0:
            if self.fps_start_timer is not None:
                total_time = timeit.default_timer() - self.fps_start_timer
                self.fps = 60 / total_time
            self.fps_start_timer = timeit.default_timer()
        self.frame_count += 1

        arcade.start_render()

        for shape in self.robots:
            shape.draw()
        self.agent.draw()
        for shape in self.start_targets:
            shape.draw()
        for shape in self.end_targets:
            shape.draw()

        # Display timings
        output = f"Processing time: {self.master.processing_time:.3f}"
        arcade.draw_text(output, 20, SCREEN_HEIGHT - 20, arcade.color.BLACK, 16)

        output = f"Drawing time: {self.draw_time:.3f}"
        arcade.draw_text(output, 20, SCREEN_HEIGHT - 40, arcade.color.BLACK, 16)

        if self.fps is not None:
            output = f"FPS: {self.fps:.0f}"
            arcade.draw_text(output, 20, SCREEN_HEIGHT - 60, arcade.color.BLACK, 16)

        if self.plot_reward:
            self.line_target.set_xdata(self.draw_x)
            self.line_target.set_ydata(self.target_y)
            self.line_robot.set_xdata(self.draw_x)
            self.line_robot.set_ydata(self.robots_y)
            self.line_overall.set_xdata(self.draw_x)
            self.line_overall.set_ydata(self.overall_y)
            plt.draw()
            plt.pause(0.001)

        self.draw_time = timeit.default_timer() - draw_start_time

    def step(self):
        arcade.pyglet.clock.tick()

        # making sure framerate is not well above 60 fps
        start_time = timeit.default_timer()
        dt = start_time - self.step_time
        self.step_time = start_time
        dt_difference = 0.0155 - dt
        if dt_difference > 0:
            time.sleep(dt_difference)

        self.switch_to()
        self.dispatch_events()
        if self.closed:
            return
        self.dispatch_event('on_draw')
        self.flip()


class App(gym.Env):
    """ Main application class. """

    def __init__(self, always_render=False, plot_reward=False, verbose=False):
        self.print = True
        self.robots = []
        self.agent = None
        self.step_return = 0
        self.agent_angle = 0
        self.agent_dx = 0
        self.agent_dy = 0
        self.agent_speed = 0
        self.start_targets = []
        self.end_targets = []
        self.cum_reward = 0
        self.always_render = always_render
        self.plot_reward = plot_reward
        self.verbose = verbose

        self.processing_time = 0
        self.step_time = timeit.default_timer()

        self.done = False
        self.step_count = 0
        self.max_steps = AGENT_MAX_STEPS

        self.rng = np.random.default_rng()

        # maxdist = np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
        # space_len = (2 + N_BOTS)
        # self.observation_space = gym.spaces.box.Box(np.array([0, 0] + [-800, -800] * (space_len-1)), np.array([800, 800] * space_len), (2 * space_len,))
        # space_len = 2
        self.observation_space = gym.spaces.box.Box(  # x-y-agent, x-y-vector-to-target, 12-distance-sensor-values
            np.array([0, 0] + [-SCREEN_WIDTH, -SCREEN_HEIGHT] + [0]*N_SENSORS),
                     np.array([SCREEN_WIDTH, SCREEN_HEIGHT]*2 + [np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)]*N_SENSORS),
                     (4 + N_SENSORS,))
        # self.observation_space = gym.spaces.box.Box(  # x-y-agent, x-y-vector-to-target, 12-distance-sensor-values
        #     np.array([0, 0] + [-SCREEN_WIDTH, -SCREEN_HEIGHT] + [0]*12),
        #              np.array([SCREEN_WIDTH, SCREEN_HEIGHT]*2 + [np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)]*12),
        #              (16,))
        # self.observation_space = gym.spaces.box.Box(  # x-y-vector-to-target, 12-distance-sensor-values
        #     np.array([-SCREEN_WIDTH, -SCREEN_HEIGHT] + [0]*12),
        #              np.array([SCREEN_WIDTH, SCREEN_HEIGHT] + [np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)]*12),
        #              (14,))
        # self.observation_space = gym.spaces.box.Box(  # dist-to-target, angle error, 12-distance-sensor-values
        #     np.array([0, -np.pi/2] + [0]*12),
        #              np.array([maxdist, np.pi/2] + [maxdist]*12),
        #              (14,))
        self.action_space = gym.spaces.box.Box(np.array([-AGENT_MAX_SPEED, -AGENT_MAX_SPEED]), np.array([AGENT_MAX_SPEED, AGENT_MAX_SPEED]), (2,))
        # self.action_space = gym.spaces.Discrete(3)  # 0: forward, 1: turn right pi/6, 2: turn left pi/6

        self.wind = None

    def setup(self):
        """ Set up the game and initialize the variables. """

        self.print = True
        self.robots = []
        self.agent = None
        self.start_targets = []
        self.end_targets = []
        self.step_count = 0
        self.done = False

        robot_rng = np.random.default_rng()

        def near_robot(x, y, robots):
            d = SCREEN_HEIGHT
            for rob in robots:
                dx = rob.x - x
                dy = rob.y - y
                d = np.sqrt(dx**2 + dy**2)

                if d <= COLLISION_DISTANCE:
                    return True

            return False

        space = (SCREEN_HEIGHT - 100) / (N_BINS - 1)

        for i in range(N_BINS):
            self.start_targets.append(Target(50, 50 + i * space))
            self.end_targets.append(Target(SCREEN_WIDTH - 50, 50 + i * space))
            # self.start_targets.append(Target(50 + i * space, 50))
            # self.end_targets.append(Target(50 + i * space, SCREEN_HEIGHT - 50))

        for i in range(N_BOTS):
            # rx = self.rng.random()
            # ry = self.rng.random()
            rx = robot_rng.random()
            ry = robot_rng.random()
            nx = rx * (SCREEN_WIDTH - 150) + 75
            ny = ry * (SCREEN_HEIGHT - 150) + 75

            while near_robot(nx, ny, self.robots):
                # rx = self.rng.random()
                # ry = self.rng.random()
                rx = robot_rng.random()
                ry = robot_rng.random()
                nx = rx * (SCREEN_WIDTH - 150) + 75
                ny = ry * (SCREEN_HEIGHT - 150) + 75

            # nx, ny = SCREEN_WIDTH/2, SCREEN_HEIGHT/2

            # ti_start = self.rng.integers(0, N_BINS-1)
            # ti_end = self.rng.integers(0, N_BINS - 1)
            ti_start = robot_rng.integers(0, N_BINS)
            ti_end = robot_rng.integers(0, N_BINS)

            self.robots.append(Robot(nx, ny, self.start_targets[ti_start], self.end_targets[ti_end]))

            # self.robots[-1].has_package = self.rng.random() < 0.5
            self.robots[-1].has_package = robot_rng.random() < 0.5

        if CENTER_START:
            nx = SCREEN_WIDTH / 2
            ny = SCREEN_HEIGHT / 2
        else:
            rx = self.rng.random()
            ry = self.rng.random()
            nx = rx * (SCREEN_WIDTH - 150) + 75
            ny = ry * (SCREEN_HEIGHT - 150) + 75

            while near_robot(nx, ny, self.robots):
                rx = self.rng.random()
                ry = self.rng.random()
                nx = rx * (SCREEN_WIDTH - 150) + 75
                ny = ry * (SCREEN_HEIGHT - 150) + 75

        if TARGET_INDEX is None:
            ti_start = self.rng.integers(0, N_BINS )
            package = self.rng.random() < 0.5
        else:
            ti_start = TARGET_INDEX % N_BINS
            package = TARGET_INDEX // N_BINS

        if package:
            targets = self.end_targets
        else:
            targets = self.start_targets

        self.agent = Agent(nx, ny, targets[ti_start], self.robots)
        self.agent.target.active = True
        self.agent.has_package = package

        if self.wind is None and self.always_render:
            self.wind = Wind(self, self.always_render, self.plot_reward)
        

    def step(self, action):
        self.step_count += 1
    
        # Ehemals update()
        """ Move everything """ 
        start_time = timeit.default_timer()
        dt = start_time - self.step_time
        if not IGNORE_ROBOTS and ROBOT_MAX_SPEED > 0:
            for r in self.robots:
                r.step(dt, self.robots)
                if r.drop_target is None:
                    r.drop_target = self.end_targets[self.rng.integers(0, N_BINS )]
                if r.pick_target is None:
                    r.pick_target = self.start_targets[self.rng.integers(0, N_BINS)]
        self.step_time = timeit.default_timer()
        signal, reward = self.agent.step(action, self.robots)
        # print('distance = {:.2f}, reward = {:.2f} * 8'.format(self.agent.robot_distances[0], reward/8))
        if signal == Agent.TARGET_FOUND:
            if self.verbose:
                print("Target Found! Reward: {}".format(REWARD_TARGET_FOUND))
            self.done = True
        if signal == Agent.ROBOT_COLLISION and DONE_AT_COLLISION:
            self.done = True
        if signal == Agent.BOUNDARY_COLLISION:
            self.done = True
        if self.agent.target is None:
            if TARGET_INDEX is None:
                ti = self.rng.integers(0, N_BINS)
            else:
                ti = TARGET_INDEX % N_BINS
            self.agent.target = self.end_targets[ti] if self.agent.has_package else self.start_targets[ti]
            self.agent.target.active = True
        self.processing_time = timeit.default_timer() - start_time

        if self.wind is not None:
            if self.wind.rendering:
                self.wind.step()
            else:
                arcade.pyglet.clock.tick()

        if self.step_count >= self.max_steps:
            if self.verbose:
                print("time's up!")
            self.done = True

        self.cum_reward += reward
        if self.done:
            if self.verbose:
                print("Cumulative Reward for this episode: {}".format(self.cum_reward))
            self.cum_reward = 0

        # return observation, reward, done, info
        rob_pos = [[r.x - self.agent.x, r.y - self.agent.y] for r in self.robots]
        # return np.array([self.agent.x, self.agent.y] +
        #     [self.agent.target.x - self.agent.x, self.agent.target.y - self.agent.y] +
        #     [coord for coord_list in rob_pos for coord in coord_list]), reward, self.done, dict()
        return np.concatenate(([self.agent.x, self.agent.y],
                        [self.agent.target.x - self.agent.x, self.agent.target.y - self.agent.y],
                        self.agent.sensor_values)), reward, self.done, dict()
        # return np.array([self.agent.dist_to_target, self.agent.angle_error] +
        #                         self.agent.sensor_values), reward, self.done, dict()

    def reset(self):
        if self.wind is not None and not self.always_render:
            self.wind.rendering = False
        self.setup()
        rob_pos = [[r.x - self.agent.x, r.y -  self.agent.y] for r in self.robots]
        # return np.array([self.agent.x, self.agent.y] +
        #                 [self.agent.target.x - self.agent.x, self.agent.target.y - self.agent.y] +
        #                 [coord for coord_list in rob_pos for coord in coord_list])
        return np.concatenate(([self.agent.x, self.agent.y],
                        [self.agent.target.x - self.agent.x, self.agent.target.y - self.agent.y],
                        self.agent.sensor_values))
        # return np.array([self.agent.dist_to_target, self.agent.angle_error] +
        #                         self.agent.sensor_values)

    def render(self, mode='human'):
        if self.wind is None:
            self.wind = Wind(self, render=True, plot_reward=self.plot_reward)
        if not self.wind.rendering:
            self.wind.rendering = True
        # if self.print:
        #     self.print = False
        #     print("Rendering is always on")


def main():
    app = App(always_render=True, plot_reward=True)
    app.setup()
    app.render()
    app.agent.x, app.agent.y = 600, 600
    app.robots[0].x, app.robots[0].y = 450, 450
    # while app.wind is None or not app.wind.closed:
    for i in range(100):
        print(i)
        app.step(np.array([-AGENT_MAX_SPEED, -AGENT_MAX_SPEED]))
        app.wind.on_update(1)
        app.wind.on_draw()
        time.sleep(2)
    # arcade.run()


if __name__ == '__main__':
    main()
