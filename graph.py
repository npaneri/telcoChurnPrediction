# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:45:39 2017

@author: nimish.paneri
"""
import graphviz as gv
g1 = gv.Graph(format='svg')
g1.node('A')
g1.node('B')
g1.edge('A', 'B')
print(g1.source)