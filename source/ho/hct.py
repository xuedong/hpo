#!/usr/bin/env python

# import random
import numpy as np
import math


class HCT(object):
    def __init__(self, support, support_type, father, depth, rho, nu, tvalue, tau, sigma, box):
        self.bvalue = float('inf')
        self.uvalue = float('inf')
        self.tvalue = tvalue
        self.tau = tau
        self.reward = 0.
        self.mean_reward = None
        self.noisy = None
        self.evaluated = None
        self.support = support
        self.support_type = support_type
        self.father = father
        self.depth = depth
        self.rho = rho
        self.nu = nu
        self.sigma = sigma
        self.box = box
        self.children = []
        self.change_status = True

    def add_children(self, c, dvalue):
        supports, supports_type = self.box.split(self.support, self.support_type, self.box.nsplits)
        # print(supports)

        tau = c**2 * math.log(1./dvalue) * self.rho**(-2*(self.depth+1))/(self.nu**2)
        self.children = [HCT(supports[i], supports_type[i],
                             self, self.depth + 1, self.rho, self.nu, 0, tau, self.sigma, self.box)
                         for i in range(len(supports))]
        # print(self.depth)

    def explore(self, c, dvalue):
        if self.tvalue < self.tau:
            self.change_status = False
            return self
        elif not self.children:
            self.add_children(c, dvalue)
            self.change_status = True
            return self.children[np.random.choice(len(self.children))]
        else:
            return max(self.children, key=lambda x: x.bvalue).explore(c, dvalue)

    def update_node(self, c, dvalue):
        if self.tvalue == 0:
            self.uvalue = float('inf')
        else:
            mean = float(self.reward)/float(self.tvalue)
            hoeffding = c * math.sqrt(math.log(1./dvalue)/float(self.tvalue))
            variation = self.nu * math.pow(self.rho, self.depth)

            self.uvalue = mean + hoeffding + variation

    def update_path(self, reward, c, dvalue):
        if not self.children:
            self.reward += reward
            self.tvalue += 1
        self.update_node(c, dvalue)

        if not self.children:
            self.bvalue = self.uvalue
        else:
            self.bvalue = min(self.uvalue, max([child.bvalue for child in self.children]))

        if self.father is not None:
            self.father.update_path(reward, c, dvalue)

    def update(self, c, dvalue):
        self.update_node(c, dvalue)
        if not self.children:
            self.bvalue = self.uvalue
        else:
            for child in self.children:
                child.update(c, dvalue)
                self.bvalue = min(self.uvalue, max([child.bvalue for child in self.children]))

    def sample(self, c, dvalue):
        leaf = self.explore(c, dvalue)
        self.change_status = leaf.change_status
        existed = False

        if leaf.noisy is None:
            x = self.box.center(leaf.support, leaf.support_type)
            leaf.evaluated = x
            leaf.mean_reward = self.box.f_mean(x)
            leaf.noisy = leaf.mean_reward + self.sigma * np.random.normal(0, self.sigma)
            existed = True
        else:
            leaf.mean_reward = self.box.f_mean(leaf.evaluated)
        leaf.update_path(leaf.noisy, c, dvalue)

        return leaf.evaluated, leaf.mean_reward, leaf.noisy, existed

    def get_change_status(self):
        return self.change_status

    def reset_change_status(self):
        self.change_status = False
