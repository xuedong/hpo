#!/usr/bin/env python

import random
import numpy as np
import math

# import source.ho.hct as hct


class PCT(object):
    def __init__(self, support, support_type, father, depth, rhos, taus, sigma, nu, box):
        self.bvalues = np.array([float("inf")] * len(rhos))
        self.uvalues = np.array([float("inf")] * len(rhos))
        self.tvalues = np.array([0] * len(rhos))
        self.rewards = np.array([0.] * len(rhos))
        self.noisy = None
        self.evaluated = None
        self.support = support
        self.support_type = support_type
        self.father = father
        self.depth = depth
        self.rhos = rhos
        self.taus = taus
        self.sigma = sigma
        self.nu = nu
        self.box = box
        self.children = []
        # self.hcts = [hct.HCT(support, support_type, father, depth, rhos[k], tau, nu, box)
        #              for k in range(len(rhos))]

    def add_children(self, c, dvalue):
        supports, supports_type = self.box.split(self.support, self.support_type, self.box.nsplits)

        taus = np.array([c ** 2 * math.log(1. / dvalue) * self.rhos[k] ** (-2 * (self.depth + 1)) / (self.nu ** 2)
                         for k in range(len(self.rhos))])
        # print(taus)
        self.children = [PCT(supports[i], supports_type[i], self, self.depth + 1, self.rhos, taus, self.sigma,
                             self.nu, self.box)
                         for i in range(len(supports))]

    def explore(self, k, c, dvalue):
        # print(self.taus)
        if self.tvalues[k] < self.taus[k]:
            return self
        elif not self.children:
            self.add_children(c, dvalue)
            return random.choice(self.children)
        else:
            return max(self.children, key=lambda x: x.bvalues[k]).explore(k, c, dvalue)

    def update_node(self, k, c, dvalue):
        if self.tvalues[k] == 0:
            self.uvalues[k] = float("inf")
        else:
            mean = float(self.rewards[k]) / float(self.tvalues[k])
            hoeffding = c * math.sqrt(math.log(1. / dvalue) / float(self.tvalues[k]))
            variation = self.nu * math.pow(self.rhos[k], self.depth)

            self.uvalues[k] = mean + hoeffding + variation

    def update_path(self, reward, k, c, dvalue):
        if not self.children:
            self.rewards[k] += reward
            self.tvalues[k] += 1
        self.update_node(k, c, dvalue)

        if not self.children:
            self.bvalues[k] = self.uvalues[k]
        else:
            self.bvalues[k] = min(self.uvalues[k], max([child.bvalues[k] for child in self.children]))

        if self.father is not None:
            self.father.update_path(reward, k, c, dvalue)

    def update_all(self, c, dvalue):
        for k in range(len(self.rhos)):
            self.update_node(k, c, dvalue)

        if not self.children:
            self.bvalues = self.uvalues
        else:
            for child in self.children:
                child.update_all(c, dvalue)
            for k in range(len(self.rhos)):
                self.bvalues[k] = min(self.uvalues[k], max([child.bvalues[k] for child in self.children]))

    def sample(self, k, c, dvalue):
        leaf = self.explore(k, c, dvalue)
        existed = False

        if leaf.noisy is None:
            x = self.box.center(leaf.support, leaf.support_type)
            leaf.evaluated = x
            leaf.mean_reward = self.box.f_mean(x)
            leaf.noisy = leaf.mean_reward + self.sigma * np.random.normal(0, self.sigma)
            existed = True
        leaf.update_path(leaf.noisy, k, c, dvalue)

        return leaf.evaluated, leaf.mean_reward, leaf.noisy, existed
