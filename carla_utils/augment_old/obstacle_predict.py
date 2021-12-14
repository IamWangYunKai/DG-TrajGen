
import matplotlib.pyplot as plt


class ObstaclePredict(object):
    def __init__(self, obstacle, dt, ds):
        self.length = obstacle.length
        self.width = obstacle.width
        self.height = obstacle.height
        self.radius = obstacle.radius
        self.id = obstacle.id

        self.dt, self.ds = dt, ds

        self.start, self.end = -1, -1
        self.time_span = 0
        self.knot_s = []

    def append(self, time_step, state, s):
        if self.time_span == 0:
            self.start = self.end = time_step
        # self.time_steps.append(time_step)
        # self.states.append(state)
        self.knot_s.append(s)
        self.time_span += 1
        self.end += 1

    def get_occupy_zone(self, time_step, expansion_radius):
        # return: S
        if time_step < self.start or time_step >= self.end:
            return None

        s = self.knot_s[time_step-self.start]
        return [s-self.radius-expansion_radius, s+self.radius+expansion_radius]


    def draw_plt(self, expansion_radius, color='-b'):
        # in S-T graph
        for time_step in range(self.start, self.end-1):
            x0 = time_step*self.dt
            s = self.get_occupy_zone(time_step, expansion_radius)
            y01 = s[0]
            y02 = s[1]

            x1 = (time_step+1)*self.dt
            s = self.get_occupy_zone(time_step+1, expansion_radius)
            y11 = s[0]
            y12 = s[1]

            plt.plot([x0,x0], [y01, y02], color)
            plt.plot([x1,x1], [y11, y12], color)

            plt.plot([x0,x1], [y01, y11], color)
            plt.plot([x0,x1], [y02, y12], color)