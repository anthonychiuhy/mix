# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:12:47 2019

@author: Anthony
"""

values = [99, 8, 5, 22, 65, 3, 43]
weights = [4, 6, 3, 76, 33, 41, 12]
states = [0, 0, 0, 0, 0, 0, 0]

capacity = 50
N = len(states)



def optval(states):
    
    K = capacity
    densities = [values[i]/weights[i] for i in range(len(values))]
    argsort = [x[0] for x in sorted(enumerate(densities), key=lambda arg:arg[1], reverse=True)]
    argones = []
    argzeros = []
    for arg in argsort:
        if states[arg] == 1:
            argones.append(arg)
        elif states[arg] == 0:
            argzeros.append(arg)
    
    
    optvalue = 0
    for arg in argones:
        optvalue += values[arg]
        K -= weights[arg]
    
    if K > 0:
        for arg in argzeros:
            if K - weights[arg] >= 0:
                optvalue += values[arg]
                K -= weights[arg]
            else:
                optvalue += densities[arg] * K
                K = 0
                break
    
    return optvalue

print(optval(states))


class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.v = val

class Tree:
    def __init__(self, N):
        self.root = Node(N * [0])
        self._buildTree(self.root)
    
    def _buildTree(self, node):
        if 0 in node.v:
            idx = node.v.index(0)
            childl = node.v.copy()
            childl[idx] = 1
            childr = node.v.copy()
            childr[idx] = -1
            
            node.l = Node(childl)
            self._buildTree(node.l)
            node.r = Node(childr)
            self._buildTree(node.r)
        else:
            return
    
    def printTree(self):
        self._printTree(self.root)
    
    def _printTree(self, node):
        if node.l is not None:
            self._printTree(node.l)
            self._printTree(node.r)
            print(node.v)
        else:
            print(node.v)


def add(n):
    if n > 1:
        return 1 + add(n-1)
    elif n == 1:
        return 1
