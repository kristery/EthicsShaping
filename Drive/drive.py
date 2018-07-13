import numpy as np

np.random.seed(1234)

class Driving(object):
    def __init__(self, num_lanes=5, p_car=0.16, p_cat=0.09, sim_len=300, ishuman_n=False, ishuman_p=False):
        self.num_lanes = num_lanes
        self.road_length = 8
        self.car_speed = 1
        self.cat_speed = 3
        self.actions = range(3)
        self.p_car = p_car
        self.p_cat = p_cat
        self.sim_len = sim_len
        self.ishuman_n = ishuman_n
        self.ishuman_p = ishuman_p

    def reset(self):
        self.lane = 2
        self.timestamp = 0
        self.done = False
        self.num_collision = 0
        self.num_hit_cat = 0
        self.cars = {}
        self.cats = {}
        for lane in range(self.num_lanes):
            self.cars[lane] = []
            self.cats[lane] = []
        # the state shows the positions of the first cat and car in adjacent lanes
        self.state_generator()
        return self.state
    
    def checker(self, lane):
        if len(self.cars[lane]) == 0:
            self.state += (-1,)
        else:
            self.state += (self.cars[lane][0],)
        if len(self.cats[lane]) == 0:
            self.state += (-1,)
        else:
            self.state += (self.cats[lane][0],)

    def state_generator(self):
        self.state = (self.lane,)
        self.checker(self.lane)
        if self.lane > 0:
            self.checker(self.lane-1)
        else:
            self.state += (-2, -2)
        if self.lane < self.num_lanes-1:
            self.checker(self.lane+1)
        else:
            self.state += (-2, -2)

    def clip(self, x):
        return min(max(x, 0), self.num_lanes-1)

    def step(self, action):
        self.timestamp += 1
        if action not in self.actions:
            raise AssertionError
        if action == 1:
            next_lane = self.clip(self.lane + 1)
        elif action == 2:
            next_lane = self.clip(self.lane - 1)
        else:
            next_lane = self.lane
        for lane in range(self.num_lanes):
            self.cats[lane] = [pos - self.cat_speed for pos in self.cats[lane]]
            self.cars[lane] = [pos - self.car_speed for pos in self.cars[lane]]

        cat_hit = 0
        car_hit = 0
        if self.lane != next_lane:
            for cat in self.cats[self.lane] + self.cats[next_lane]:
                if cat <= 0: cat_hit += 1
            for car in self.cars[self.lane] + self.cars[next_lane]:
                if car <= 0: car_hit += 1
            self.lane = next_lane
        else:
            for cat in self.cats[self.lane]:
                if cat <= 0: cat_hit += 1
            for car in self.cars[self.lane]:
                if car <= 0: car_hit += 1

        for lane in range(self.num_lanes):
            self.cats[lane] = [pos for pos in self.cats[lane] if pos > 0]
            self.cars[lane] = [pos for pos in self.cars[lane] if pos > 0]

        if np.random.rand() < self.p_car:
            self.cars[np.random.randint(5)].append(self.road_length)
        if np.random.rand() < self.p_cat:
            self.cats[np.random.randint(5)].append(self.road_length)

        if self.ishuman_n:
            reward = -20 * cat_hit + -1 * car_hit + 0.5 * (action == 0)
        elif self.ishuman_p:
            reward = 20 * cat_hit + -1 * car_hit + 0.5 * (action == 0)
        else:
            reward = -20 * car_hit + 0.5 * (action == 0)
        
        self.num_collision += car_hit
        self.num_hit_cat += cat_hit
        if self.timestamp >= self.sim_len:
            self.done = True

        self.state_generator()
        return self.state, reward, self.done 

    def log(self):
        return self.num_collision, self.num_hit_cat





