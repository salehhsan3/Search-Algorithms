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

    def h_SAP(self, tmp_env: FrozenLakeEnv, goal_node: int,state: int):
        state_row , state_column = tmp_env.to_row_col(state)
        goal_row, goal_column = tmp_env.to_row_col(goal_node)
        costs = {b"F": 10.0, b"H": np.inf, b"T": 3.0, b"A": 2.0, b"L": 1.0, b"S": 1.0, b"G": 1.0, b"P": 100}
        return min(costs[b"P"],abs(goal_row-state_row)+abs(goal_column-state_column))


    def h_MSAP(self, tmp_env: FrozenLakeEnv, G: List, state: int):
        min = np.inf
        for g in G:
            value = self.h_SAP(tmp_env,g,state)
            if value < min:
                min = value
        return min



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
                if terminated and not env.is_final_state(next_state):
                    continue
                if (next_state not in self.OPEN_SET) and (next_state not in self.CLOSE):
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


class GreedyAgent(Agent):
  
    def __init__(self) -> None:
        self.OPEN = queue.PriorityQueue()
        super().__init__()

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        initial_node = GraphNode(env.s, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([], 0, 0)
        self.OPEN.put((self.h_MSAP((env,env.goals,initial_node.state)), initial_node))
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand_idx, node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.CLOSE[node_to_expand.state] = node_to_expand
            self.expanded_counter += 1

            for act, tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                if (next_state not in self.OPEN_SET) and (next_state not in self.CLOSE):
                    new_node = GraphNode(next_state, node_to_expand, act, cost)
                    if env.is_final_state(new_node.state):
                        return self.get_path(new_node, self.expanded_counter)
                    elif terminated and not env.is_final_state(new_node.state):
                        continue
                    new_node.h_value = node_to_expand.h_value
                    self.OPEN.put( (self.h_MSAP((env,env.goals,new_node.state)) , new_node) )
                    self.OPEN_SET[next_state] = new_node

class WeightedAStarAgent(Agent):
    
    def __init__(self):
        self.OPEN = queue.PriorityQueue()
        super().__init__()

    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], int, float]:
        initial_node = GraphNode(env.s, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([], 0, 0)
        self.OPEN.put((initial_node.g_value, initial_node))
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand_idx, node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.CLOSE[node_to_expand.state] = node_to_expand
            if env.is_final_state(node_to_expand.state):
                return self.get_path(node_to_expand, self.expanded_counter)
            self.expanded_counter += 1

            for act, tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                if (next_state not in self.OPEN_SET) or (next_state not in self.CLOSE):
                    new_node = GraphNode(next_state, node_to_expand, act, cost)
                    new_node.g_value = node_to_expand.g_value + cost
                    self.OPEN.put((new_node.g_value, new_node))
                    self.OPEN_SET[next_state] = new_node
                elif next_state in self.OPEN_SET:
                    next_node = self.OPEN_SET[next_state]
                    new_cost = node_to_expand.g_value + cost
                    if new_cost < next_node.g_value:
                        next_node.g_value = new_cost
                        next_node.parent = node_to_expand
                        next_node.action = act
                        next_node.cost = cost


class IDAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError