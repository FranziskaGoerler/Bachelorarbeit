import arcade
import gym
import random
import timeit
import math
import numpy as np
import time

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
N_BINS = 4
N_BOTS = 1
COLLISION_DISTANCE = 35
COLLISION_DISTANCE = 150+17.5
PLAN_STEPS = 20         # default 45
CHECK_MAX_STEPS = 5    # default 15
RADIUS = 100
MIN_DIST = 20
SCREEN_TITLE = "pycking environment"

ROBOT_MAX_SPEED = 0    # if exactly 0, step function of robot is never called, which speeds up environment
AGENT_MAX_SPEED = 8
PUNISH_WRONG_DIRECTION = False
REWARD_TARGET_FOUND = 5000    # reward for reaching target
REWARD_COLLISION = -5000
REWARD_BOUNDARY = -5000
DONE_AT_COLLISION = True
ONLY_NEAREST_ROBOT = True
IGNORE_ROBOTS = False
AGENT_MAX_STEPS = 350    # max length of an episode
TARGET_INDEX = None      # Index of target for agent. None for random target each episode.
CENTER_START = False   # Whether to always put agent in the screen center (True) or initialize randomly (False)

# DESCRIPTION = ''
DESCRIPTION = 'This has one big robot (radius 150 instead of 35) in the middle, as a static obstacle. The robot position is not part of the observation. Training is based on ppo_sandbox/2/agent.'

param_names = ['SCREEN_WIDTH', 'SCREEN_HEIGHT', 'N_BINS', 'N_BOTS', 'COLLISION_DISTANCE', 'PLAN_STEPS', 'CHECK_MAX_STEPS', 'RADIUS', 'MIN_DIST', 'SCREEN_TITLE', 'ROBOT_MAX_SPEED',
'AGENT_MAX_SPEED', 'PUNISH_WRONG_DIRECTION', 'REWARD_TARGET_FOUND', 'REWARD_COLLISION', 'REWARD_BOUNDARY', 'DONE_AT_COLLISION', 'ONLY_NEAREST_ROBOT',
               'IGNORE_ROBOTS', 'AGENT_MAX_STEPS', 'TARGET_INDEX', 'CENTER_START', 'DESCRIPTION']

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
        self._radius = 17.5
        self._radius = 150
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

        point_list = ((self.x, self.y), (self.x + 10, self.y + 10), (self.x + 14, self.y + 18))
        self.planned_path = Line(point_list)

    def step(self, dt, robots):
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
        self.shape_list = arcade.ShapeElementList()
        self.shape_list.append(shape)


class Agent:
    OK = 0
    TARGET_FOUND = 1
    BOUNDARY_COLLISION = 2
    ROBOT_COLLISION = 3

    def __init__(self, xpos, ypos, first_target):
        self.x = xpos
        self.y = ypos
        self.has_package = False
        self.target = first_target
        self._radius = 17.5

    def step(self, action, robots):

        def near_robot(x, y, robots):
            d = SCREEN_HEIGHT
            for rob in robots:
                dx = rob.x - x
                dy = rob.y - y
                d = np.sqrt(dx**2 + dy**2)

                if d <= COLLISION_DISTANCE:
                    return True

            return False

        action_length = np.sqrt(np.sum(action ** 2))
        if action_length > AGENT_MAX_SPEED:
            action = (action / action_length) * AGENT_MAX_SPEED
            action_length = AGENT_MAX_SPEED

        nx = self.x + action[0]
        ny = self.y + action[1]

        pre_diffx, pre_diffy = self.target.x - self.x, self.target.y - self.y

        if not IGNORE_ROBOTS and near_robot(nx, ny, robots):
            return Agent.ROBOT_COLLISION, REWARD_COLLISION
        if nx > SCREEN_WIDTH or nx < 0 or ny > SCREEN_HEIGHT or ny < 0:
            return Agent.BOUNDARY_COLLISION, REWARD_BOUNDARY
        self.x = nx
        self.y = ny

        diffx, diffy = self.target.x - self.x, self.target.y - self.y

        if abs(diffx) <= MIN_DIST and abs(diffy) <= MIN_DIST:
            # target reached
            self.has_package = not self.has_package
            self.target.active = False
            self.target = None
            return Agent.TARGET_FOUND, REWARD_TARGET_FOUND

        angle_to_destination = np.arctan2(pre_diffy, pre_diffx) # Wäre der bestmögliche Winkel gewesen
        action_angle = np.arctan2(action[1], action[0])
        angle_error = angle_to_destination - action_angle
        reward_factor = (1 / (np.exp(np.pi / 2) - 1)) * action_length
        reward = 0
        # (2) Belohnung abhängig vom Verhältnis des gewählten Winkels zum perfekten Winkel
        if angle_error <= 0 and angle_error >= - np.pi / 2:
            reward = (np.exp(angle_error + np.pi / 2) - 1) * reward_factor
        elif angle_error > 0 and angle_error <= np.pi / 2:
            reward = (np.exp(- angle_error + np.pi / 2) - 1) * reward_factor
        elif PUNISH_WRONG_DIRECTION:
            reward = - action_length * ((2 / np.pi) * abs(angle_error) - 1)

        return Agent.OK, reward


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
    def __init__(self, master, render=False):
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

        self.rendering = render
        self.closed = False

        @self.event
        def on_close():
            # print("I'm closing now")
            self.closed = True
            self.close()

    def on_update(self, dt):

        def color_check(t, tS):
            if t.active and tS.color == arcade.color.MAGENTA:
                tS.change_color(arcade.color.RED_ORANGE)
            elif not t.active and tS.color == arcade.color.RED_ORANGE:
                tS.change_color(arcade.color.MAGENTA)

        self.agent.x = self.master.agent.x
        self.agent.y = self.master.agent.y
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

    def __init__(self, always_render=False, verbose=False):
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
        self.verbose = verbose

        self.processing_time = 0
        self.step_time = timeit.default_timer()

        self.done = False
        self.step_count = 0
        self.max_steps = AGENT_MAX_STEPS

        self.rng = np.random.default_rng()

        space_len = (2 + N_BOTS) if not ONLY_NEAREST_ROBOT else 3
        space_len = 2
        self.observation_space = gym.spaces.box.Box(np.array([0, 0] + [-800, -800] * (space_len-1)), np.array([800, 800] * space_len), (2 * space_len,))
        self.action_space = gym.spaces.box.Box(np.array([-AGENT_MAX_SPEED, -AGENT_MAX_SPEED]), np.array([AGENT_MAX_SPEED, AGENT_MAX_SPEED]), (2,))

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

            nx, ny = SCREEN_WIDTH/2, SCREEN_HEIGHT/2

            # ti_start = self.rng.integers(0, N_BINS-1)
            # ti_end = self.rng.integers(0, N_BINS - 1)
            ti_start = robot_rng.integers(0, N_BINS-1)
            ti_end = robot_rng.integers(0, N_BINS - 1)

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
            ti_start = self.rng.integers(0, N_BINS - 1)
            package = self.rng.random() < 0.5
        else:
            ti_start = TARGET_INDEX % N_BINS
            package = TARGET_INDEX // N_BINS

        if package:
            targets = self.end_targets
        else:
            targets = self.start_targets

        self.agent = Agent(nx, ny, targets[ti_start])
        self.agent.target.active = True
        self.agent.has_package = package

        if self.wind is None and self.always_render:
            self.wind = Wind(self, self.always_render)
        

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
                    r.drop_target = self.end_targets[self.rng.integers(0, N_BINS - 1)]
                if r.pick_target is None:
                    r.pick_target = self.start_targets[self.rng.integers(0, N_BINS - 1)]
        self.step_time = timeit.default_timer()
        signal, reward = self.agent.step(action, self.robots)
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
                ti = self.rng.integers(0, N_BINS - 1)
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
        if ONLY_NEAREST_ROBOT:
            rob_distance = [np.sqrt(r[0]**2 + r[1] ** 2) for r in rob_pos]
            rob_pos = [rob_pos[np.argmin(rob_distance)]]
        rob_pos = []
        return np.array([self.agent.x, self.agent.y] + [self.agent.target.x - self.agent.x, self.agent.target.y - self.agent.y] + [coord for coord_list in rob_pos for coord in coord_list]), reward, self.done, dict()

    def reset(self):
        if self.wind is not None and not self.always_render:
            self.wind.rendering = False
        self.setup()
        rob_pos = [[r.x - self.agent.x, r.y -  self.agent.y] for r in self.robots]
        if ONLY_NEAREST_ROBOT:
            rob_distance = [np.sqrt(r[0]**2 + r[1] ** 2) for r in rob_pos]
            rob_pos = [rob_pos[np.argmin(rob_distance)]]
        rob_pos = []
        return np.array([self.agent.x, self.agent.y] + [self.agent.target.x - self.agent.x, self.agent.target.y - self.agent.y] + [coord for coord_list in rob_pos for coord in coord_list])

    def render(self, mode='human'):
        if self.wind is None:
            self.wind = Wind(self, True)
        if not self.wind.rendering:
            self.wind.rendering = True
        # if self.print:
        #     self.print = False
        #     print("Rendering is always on")


def main():
    app = App()
    app.setup()
    app.render()
    while app.wind is None or not app.wind.closed:
        app.step([0, 1])
    # arcade.run()


if __name__ == '__main__':
    main()
