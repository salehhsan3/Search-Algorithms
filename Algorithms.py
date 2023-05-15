import numpy as np

from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict
import queue

class GraphNode:
    def __init__(self,state,parent,action,cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.g_value = 0
        self.h_value = 0
        self.f_value = 0
        self.weight = 0

class Agent:
    def __init__(self) -> None:
        self.OPEN_SET = {}
        self.CLOSE = {}
        self.expanded_counter = 0.0
    def get_path(self,goal_node: GraphNode,expanded) -> Tuple[List[int] ,int, float]:
        lst = []
        node_iter = goal_node
        cost = 0
        while node_iter.parent is not None:
            lst.insert(0,node_iter.action)
            cost += node_iter.cost
        return (lst,cost,expanded)

    def h_MSAP(self,G: set, state: int):
        env_ = FrozenLakeEnv()
        state_x, state_y = env_.to_row_col()
        Manhattan_dist = env_.inf
        for g in G:
            goal_x, goal_y = env_.to_row_col()


class BFSAgent(Agent):
    def __init__(self) -> None:
        self.OPEN = queue.SimpleQueue()
        super().__init__()

       # self.OPEN_SET = set()
       # self.CLOSE = set()
       # self.expanded_counter = 0.0

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        initial_node = GraphNode(env.s,None,None,None)
        if env.is_final_state(initial_node.state):
            return ([],0,0)
        self.OPEN.put(initial_node)
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.expanded_counter += 1
            self.CLOSE[node_to_expand.state] = node_to_expand

            for act,tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                if terminated:
                    if env.is_final_state(next_state):
                        final_node = GraphNode(next_state, node_to_expand, act, cost)
                        return self.get_path(final_node,self.expanded_counter)
                    else:
                        pass # this is A HOLE
                if (next_state in self.OPEN_SET) or (next_state in self.CLOSE):
                    pass
                else:
                    new_node = GraphNode(next_state,node_to_expand,act,cost)
                    self.OPEN.put(new_node)
                    self.OPEN_SET[next_state] = new_node




class DFSAgent():
    def __init__(self) -> None:
        self.OPEN = queue.LifoQueue()
        super().__init__()

    # self.OPEN_SET = set()
    # self.CLOSE = set()
    # self.expanded_counter = 0.0

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        initial_node = GraphNode(env.s, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([],0,0)
        self.OPEN.put(initial_node)
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.expanded_counter += 1
            self.CLOSE[node_to_expand.state] = node_to_expand

            for act, tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                if terminated:
                    if env.is_final_state(next_state):
                        final_node = GraphNode(next_state, node_to_expand, act, cost)
                        return self.get_path(final_node, self.expanded_counter)
                    else:
                        pass  # this is A HOLE
                if (next_state in self.OPEN_SET) or (next_state in self.CLOSE):
                    pass
                else:
                    new_node = GraphNode(next_state, node_to_expand, act, cost)
                    self.OPEN.put(new_node)
                    self.OPEN_SET[next_state] = new_node


class UCSAgent(Agent):

    def __init__(self) -> None:
        self.OPEN = queue.PriorityQueue()
        super().__init__()

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        initial_node = GraphNode(env.s, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([], 0, 0)
        self.OPEN.put((initial_node.g_value,initial_node))
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand_idx, node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.CLOSE[node_to_expand.state] = node_to_expand
            if env.is_final_state(node_to_expand.state):
                return self.get_path(node_to_expand,self.expanded_counter)
            self.expanded_counter += 1

            for act, tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                 if (next_state not in self.OPEN_SET) or (next_state not in self.CLOSE):
                     new_node = GraphNode(next_state,node_to_expand,act,cost)
                     new_node.g_value = node_to_expand.g_value + cost
                     self.OPEN.put((new_node.g_value,new_node))
                     self.OPEN_SET[next_state] = new_node
                 elif next_state in self.OPEN_SET:
                     next_node = self.OPEN_SET[next_state]
                     new_cost = node_to_expand.g_value + cost
                     if new_cost < next_node.g_value:
                         next_node.g_value = new_cost
                         next_node.parent = node_to_expand
                         next_node.action = act
                         next_node.cost = cost

class GreedyAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError

class WeightedAStarAgent(Agent):
    
    def __init__(self) -> None:
        self.OPEN = queue.PriorityQueue()
        super().__init__()

    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], int, float]:
        initial_node = GraphNode(env.s, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([], 0, 0)
        self.OPEN.put((initial_node.g_value,initial_node))
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand_idx, node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.CLOSE[node_to_expand.state] = node_to_expand
            if env.is_final_state(node_to_expand.state):
                return self.get_path(node_to_expand,self.expanded_counter)
            self.expanded_counter += 1

            for act, tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                # new_g_value = node_to_expand.g_value + cost
                # new_f_value = (1)*new_g_value + (h_weight)*self.h_MSAP(next_state)
                # self.OPEN_SET[next_state].f_value = new_f_value ## this info doesn't exist
                if (next_state not in self.OPEN_SET) and (next_state not in self.CLOSE):
                    new_node = GraphNode(next_state,node_to_expand,act,cost)
                    new_node.g_value = node_to_expand.g_value + cost
                    new_node.f_value = (1)*new_node.g_value + (h_weight)*self.h_MSAP(next_state)
                    self.OPEN.put((new_node.f_value,new_node))
                    self.OPEN_SET[next_state] = new_node
                elif (next_state in self.OPEN_SET):
                    curr_node = self.OPEN_SET[next_state]
                    new_g_value = (node_to_expand.g_value + cost) # is this correct?
                    new_f_value = (1)*new_g_value + (h_weight)*self.h_MSAP(next_state)
                    if new_f_value < curr_node.f_value:
                        curr_node.g_value = new_g_value
                        curr_node.f_value = new_f_value
                        self.OPEN.get() # essentially removes minimum element from PriorityQueue
                        self.OPEN.put((curr_node.f_value,curr_node))
                else: # next_state is in CLOSED
                    curr_node = self.CLOSE[next_state]
                    new_g_value = (node_to_expand.g_value + cost) # is this correct?
                    new_f_value = (1)*new_g_value + (h_weight)*self.h_MSAP(next_state)
                    if new_f_value < curr_node.f_value:
                        curr_node.g_value = new_g_value
                        curr_node.f_value = new_f_value
                        self.OPEN.put((curr_node.f_value,curr_node))
                        self.OPEN_SET[next_state] = curr_node
                        del self.CLOSE[next_state]


class IDAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError