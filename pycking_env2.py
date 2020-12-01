import arcade
import gym
import random
import timeit
import math
import numpy as np

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
N_BINS = 4
N_BOTS = 0
COLLISION_DISTANCE = 35
PLAN_STEPS = 20         # default 45
CHECK_MAX_STEPS = 5    # default 15
RADIUS = 100
MIN_DIST = 20
SCREEN_TITLE = "pycking environment"

AGENT_MAX_SPEED = 3
AGENT_MIN_SPEED = 1.5
AGENT_MAX_ANGLE = np.pi
AGENT_MIN_ANGLE = -np.pi
REWARD_TARGET_FOUND = 5000    # reward for reaching target
REWARD_DISTANCE_COVERED = 1   # reward for moving straight towards target with maximal speed
REWARD_COLLISION = -500
REWARD_STATIONARY = -10
DONE_AT_COLLISION = True
AGENT_MAX_STEPS = 800    # max length of an episode
TARGET_INDEX = 4      # Index of target for agent. None for random target each episode.
CENTER_START = False   # Whether to always put agent in the screen center (True) or initialize randomly (False)

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
                sp = 3.0
                if np.sqrt(diffx**2+diffy**2) < RADIUS:
                    sp = 1.5
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
    def __init__(self, xpos, ypos, first_target):
        self.x = xpos
        self.y = ypos
        self.has_package = False
        self.target = first_target
        self._radius = 17.5

    def step(self, dx, dy, speed, robots):

        def near_robot(x, y, robots):
            d = SCREEN_HEIGHT
            for rob in robots:
                dx = rob.x - x
                dy = rob.y - y
                d = np.sqrt(dx**2 + dy**2)

                if d <= COLLISION_DISTANCE:
                    return True

            return False

        if abs(dx) + abs(dy) < 10e-4:
            if REWARD_STATIONARY != 0:
                return REWARD_STATIONARY
            speed_fac = 0
        else:
            speed_fac = speed / np.sqrt(dx**2 + dy**2)
        nx = self.x + dx * speed_fac
        ny = self.y + dy * speed_fac

        target = self.target if self.has_package else self.target
        dist_to_target_before = np.sqrt((target.x - self.x)**2 + (target.y - self.y)**2)

        if near_robot(nx, ny, robots):
            return REWARD_COLLISION
        if nx > SCREEN_WIDTH or nx < 0 or ny > SCREEN_HEIGHT or ny < 0:
            return REWARD_COLLISION
        self.x = nx
        self.y = ny

        diffx, diffy = target.x - self.x, target.y - self.y
        dist_to_target_after = np.sqrt(diffx ** 2 + diffy ** 2)

        if abs(diffx) <= MIN_DIST and abs(diffy) <= MIN_DIST:
            # target reached
            self.has_package = not self.has_package
            self.target.active = False
            self.target = None
            print("Target Found! Reward: {}".format(REWARD_TARGET_FOUND))
            return REWARD_TARGET_FOUND

        distance_reward = max(0, (dist_to_target_before - dist_to_target_after) / (AGENT_MAX_SPEED/REWARD_DISTANCE_COVERED))   # reward of 1 if moving straight towards target with max speed

        return distance_reward


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

        self.switch_to()
        self.dispatch_events()
        if self.closed:
            return
        self.dispatch_event('on_draw')
        self.flip()


class App(gym.Env):
    """ Main application class. """

    def __init__(self, always_render=False):
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

        self.processing_time = 0
        self.step_time = timeit.default_timer()

        self.done = False
        self.step_count = 0
        self.max_steps = AGENT_MAX_STEPS

        space_len = (2 + N_BOTS)
        self.observation_space = gym.spaces.box.Box(np.array([0, 0]*space_len), np.array([SCREEN_WIDTH, SCREEN_HEIGHT]*space_len), (2*space_len,))
        self.action_space = gym.spaces.box.Box(np.array([-1, -1, AGENT_MIN_SPEED]), np.array([1, 1, AGENT_MAX_SPEED]), (3,))

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

        for i in range(N_BOTS):
            rx = random.random()
            ry = random.random()
            nx = rx * (SCREEN_WIDTH - 150) + 75
            ny = ry * (SCREEN_HEIGHT - 150) + 75

            while near_robot(nx, ny, self.robots):
                rx = random.random()
                ry = random.random()
                nx = rx * (SCREEN_WIDTH - 150) + 75
                ny = ry * (SCREEN_HEIGHT - 150) + 75

            ti_start = random.randint(0, N_BINS-1)
            ti_end = random.randint(0, N_BINS - 1)

            self.robots.append(Robot(nx, ny, self.start_targets[ti_start], self.end_targets[ti_end]))

        if CENTER_START:
            nx = SCREEN_WIDTH / 2
            ny = SCREEN_HEIGHT / 2
        else:
            rx = random.random()
            ry = random.random()
            nx = rx * (SCREEN_WIDTH - 150) + 75
            ny = ry * (SCREEN_HEIGHT - 150) + 75

            while near_robot(nx, ny, self.robots):
                rx = random.random()
                ry = random.random()
                nx = rx * (SCREEN_WIDTH - 150) + 75
                ny = ry * (SCREEN_HEIGHT - 150) + 75

        if TARGET_INDEX is None:
            ti_start = random.randint(0, N_BINS - 1)
            package = random.random() < 0.5
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

    def update(self):
        """ Move everything """
        start_time = timeit.default_timer()
        dt = start_time - self.step_time
        for r in self.robots:
            r.step(dt, self.robots)
            if r.drop_target is None:
                r.drop_target = self.end_targets[random.randint(0, N_BINS-1)]
            if r.pick_target is None:
                r.pick_target = self.start_targets[random.randint(0, N_BINS - 1)]
        self.step_time = timeit.default_timer()
        self.step_return = self.agent.step(self.agent_dx, self.agent_dy, self.agent_speed, self.robots)
        if self.step_return == REWARD_TARGET_FOUND:
            self.done = True
        if self.step_return == REWARD_COLLISION and DONE_AT_COLLISION:
            self.done = True
        if self.agent.target is None:
            if TARGET_INDEX is None:
                ti = random.randint(0, N_BINS - 1)
            else:
                ti = TARGET_INDEX % N_BINS
            self.agent.target = self.end_targets[ti] if self.agent.has_package else self.start_targets[ti]
            self.agent.target.active = True
        self.processing_time = timeit.default_timer() - start_time

    def step(self, action):
        self.step_count += 1
        # print(action)
        # self.agent_angle = action[0]
        self.agent_dx = action[0]
        self.agent_dy = action[1]
        self.agent_speed = action[2]

        self.update()

        if self.wind is not None:
            if self.wind.rendering:
                self.wind.step()
            else:
                arcade.pyglet.clock.tick()

        if self.step_count >= self.max_steps:
            print("time's up!")
            self.done = True

        self.cum_reward += self.step_return
        if self.done:
            print("Cumulative Reward for this episode: {}".format(self.cum_reward))
            self.cum_reward = 0

        # return observation, reward, done, info
        rob_pos = [[r.x, r.y] for r in self.robots]
        return np.array([self.agent.x, self.agent.y] + [self.agent.target.x, self.agent.target.y] + [coord for coord_list in rob_pos for coord in coord_list]), self.step_return, self.done, dict()

    def reset(self):
        if self.wind is not None and not self.always_render:
            self.wind.rendering = False
        self.setup()
        rob_pos = [[r.x, r.y] for r in self.robots]
        return np.array([self.agent.x, self.agent.y] + [self.agent.target.x, self.agent.target.y] + [coord for coord_list in rob_pos for coord in coord_list])

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
