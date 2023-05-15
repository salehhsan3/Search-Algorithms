import numpy as np

from FrozenLakeEnv import FrozenLakeEnv
from typing import List,  Tuple
import heapdict
import queue
from queue import PriorityQueue

class GraphNode:
    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.g_value = 0
        self.h_value = 0
        self.weight = 0
    
    def __eq__(self, other):
        return (self.state == other.state) and (self.parent == other.parent) and (self.action == other.action) and (self.cost == other.cost)

    def __ne__(self, other):
        return not (self == other)

    
class Agent:
    def __init__(self) -> None:
        self.OPEN_SET = {}
        self.CLOSE = {}
        self.expanded_counter = 0.0
        
    def get_path(self,  goal_node: GraphNode,  expanded) -> Tuple[List[int] , int,  float]:
        lst = []
        node_iter = goal_node
        cost = 0
        while node_iter.parent is not None:
            lst.insert(0, node_iter.action)
            cost += node_iter.cost
            node_iter = node_iter.parent
        return (lst, cost, expanded)

    def h_SAP(self,  tmp_env: FrozenLakeEnv,  goal_node: int, state: int):
        state_row, state_column = tmp_env.to_row_col(state)
        goal_row,  goal_column = tmp_env.to_row_col(goal_node)
        costs = {b"F": 10.0,  b"H": np.inf,  b"T": 3.0,  b"A": 2.0,  b"L": 1.0,  b"S": 1.0,  b"G": 1.0,  b"P": 100}
        return min(costs[b"P"], abs(goal_row-state_row)+abs(goal_column-state_column))


    def h_MSAP(self,  tmp_env: FrozenLakeEnv,  G: List,  state: int):
        min = np.inf
        for g in G:
            value = self.h_SAP(tmp_env, g, state)
            if value < min:
                min = value
        return min


def updateKey(open_queue: PriorityQueue,  open_set: dict, 
    node_to_update: GraphNode ,  old_key: int,  new_key: int) -> PriorityQueue():
    new_queue = queue.PriorityQueue()()
    for key,  node in open_set.items():
        if node is node_to_update and key == old_key:
            new_queue.put((new_key,  node))
        else:
            new_queue.put((key,  node))
    return new_queue
        
class BFSAgent(Agent):
    def __init__(self) -> None:
        self.OPEN = queue.SimpleQueue()
        super().__init__()

    def search(self,  env: FrozenLakeEnv) -> Tuple[List[int],  int,  float]:
        initial_node = GraphNode(env.reset(), None, None, None)
        if env.is_final_state(initial_node.state):
            return ([], 0, 0)
        self.OPEN.put(initial_node)
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.expanded_counter += 1
            self.CLOSE[node_to_expand.state] = node_to_expand
            if None not in env.succ(node_to_expand.state)[0]:
                for act, tup in env.succ(node_to_expand.state).items():
                    next_state,  cost,  terminated = tup
                    # print(str(env.succ(node_to_expand.state).items()))
                    # print("next-state is " + str(next_state) + " and cost is :" + str(cost) + " and terminated is " + str(terminated) )
                    # print((str(tup in env.succ(node_to_expand.state))))
                    if terminated:
                        if env.is_final_state(next_state):
                            final_node = GraphNode(next_state,  node_to_expand,  act,  cost)
                            return self.get_path(final_node, self.expanded_counter)
                        else:
                            pass # this is A HOLE
                    if (next_state in self.OPEN_SET) or (next_state in self.CLOSE):
                        pass
                    else:
                        new_node = GraphNode(next_state, node_to_expand, act, cost)
                        self.OPEN.put(new_node)
                        self.OPEN_SET[next_state] = new_node




class DFSAgent(Agent):
    def __init__(self) -> None:
        self.OPEN = queue.LifoQueue()
        super().__init__()

    def search(self,  env: FrozenLakeEnv) -> Tuple[List[int],  int,  float]:
        initial_node = GraphNode(env.reset(),  None,  None,  None)
        if env.is_final_state(initial_node.state):
            return ([], 0, 0)
        self.OPEN.put(initial_node)
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.expanded_counter += 1
            self.CLOSE[node_to_expand.state] = node_to_expand
            if None in env.succ(node_to_expand.state)[0]:   # we're expanding a Hole, so we have to return
                return ([], np.inf, self.expanded_counter) # to check if this is a valid return as "failure"!!!!!!!!!!!!!!!!!!!!!
            for act,  tup in env.succ(node_to_expand.state).items():
                next_state,  cost,  terminated = tup
                if terminated:
                    if env.is_final_state(next_state):
                        final_node = GraphNode(next_state,  node_to_expand,  act,  cost)
                        return self.get_path(final_node,  self.expanded_counter)
                    else:
                        pass  # this is A HOLE don't insert it to the OPEN list
                if (next_state in self.OPEN_SET) or (next_state in self.CLOSE):
                    pass
                else:
                    new_node = GraphNode(next_state,  node_to_expand,  act,  cost)
                    self.OPEN.put(new_node)
                    self.OPEN_SET[next_state] = new_node


class UCSAgent(Agent):

    def __init__(self) -> None:
        # self.OPEN = PriorityQueue()
        self.OPEN = heapdict.heapdict() # essentially a priority queue for our purposes
        super().__init__()

    def search(self,  env: FrozenLakeEnv) -> Tuple[List[int],  int,  float]:
        initial_node = GraphNode(env.reset(),  None,  None,  None)
        if env.is_final_state(initial_node.state):
            return ([],  0,  0)
        # self.OPEN.put((initial_node.g_value, initial_node))
        self.OPEN[initial_node.g_value] = initial_node
        self.OPEN_SET[initial_node.state] = initial_node

        while bool(self.OPEN): # meaning heap is not empty
            # node_to_expand_idx,  node_to_expand = self.OPEN.get()
            node_to_expand_idx,  node_to_expand = self.OPEN.popitem()
            del self.OPEN_SET[node_to_expand.state]
            self.CLOSE[node_to_expand.state] = node_to_expand
            if env.is_final_state(node_to_expand.state):
                return self.get_path(node_to_expand, self.expanded_counter)
            self.expanded_counter += 1
            
            if None in env.succ(node_to_expand.state)[0]:   # we're expanding a Hole, so we have to return
                return ([], np.inf, self.expanded_counter) # to check if this is a valid return as "failure"!!!!!!!!!!!!!!!!!!!!!
            for act,  tup in env.succ(node_to_expand.state).items():
                next_state,  cost,  terminated = tup
                if terminated and not env.is_final_state(next_state):
                    continue
                if (next_state not in self.OPEN_SET) and (next_state not in self.CLOSE):
                    new_node = GraphNode(next_state, node_to_expand, act, cost)
                    new_node.g_value = node_to_expand.g_value + cost
                    #  self.OPEN.put((new_node.g_value, new_node))
                    self.OPEN[new_node.g_value] = new_node
                    self.OPEN_SET[next_state] = new_node
                elif next_state in self.OPEN_SET:
                     next_node = self.OPEN_SET[next_state]
                     new_cost = node_to_expand.g_value + cost
                     if new_cost < next_node.g_value:
                         next_node.g_value = new_cost
                         next_node.parent = node_to_expand
                         next_node.action = act
                         next_node.cost = cost
        return ([], -1, -1)


class GreedyAgent(Agent):
  
    def __init__(self) -> None:
        self.OPEN = queue.PriorityQueue()
        super().__init__()

    def search(self,  env: FrozenLakeEnv) -> Tuple[List[int],  int,  float]:
        initial_node = GraphNode(env.reset(),  None,  None,  None)
        if env.is_final_state(initial_node.state):
            return ([],  0,  0)
        self.OPEN.put((self.h_MSAP((env, env.goals, initial_node.state)),  initial_node))
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand_idx,  node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.CLOSE[node_to_expand.state] = node_to_expand
            self.expanded_counter += 1

            for act,  tup in env.succ(node_to_expand.state).items():
                next_state,  cost,  terminated = tup
                if (next_state not in self.OPEN_SET) and (next_state not in self.CLOSE):
                    new_node = GraphNode(next_state,  node_to_expand,  act,  cost)
                    if env.is_final_state(new_node.state):
                        return self.get_path(new_node,  self.expanded_counter)
                    elif terminated and not env.is_final_state(new_node.state):
                        continue
                    new_node.h_value = node_to_expand.h_value
                    self.OPEN.put( (self.h_MSAP((env, env.goals, new_node.state)) ,  new_node) )
                    self.OPEN_SET[next_state] = new_node

class WeightedAStarAgent(Agent):
    
    def __init__(self) -> None:
        self.OPEN = queue.PriorityQueue()
        super().__init__()

    def search(self,  env: FrozenLakeEnv,  h_weight) -> Tuple[List[int],  int,  float]:
        initial_node = GraphNode(env.reset(),  None,  None,  None)
        Goals = env.goals
        if env.is_final_state(initial_node.state):
            return ([],  0,  0)
        self.OPEN.put((initial_node.g_value, initial_node))
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand_idx,  node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.CLOSE[node_to_expand.state] = node_to_expand
            if env.is_final_state(node_to_expand.state):
                return self.get_path(node_to_expand, self.expanded_counter)
            self.expanded_counter += 1

            for act,  tup in env.succ(node_to_expand.state).items():
                next_state,  cost,  terminated = tup
                if (next_state not in self.OPEN_SET) and (next_state not in self.CLOSE):
                    new_node = GraphNode(next_state, node_to_expand, act, cost)
                    new_node.g_value = node_to_expand.g_value + cost
                    new_node.f_value = (1)*new_node.g_value + (h_weight)*self.h_MSAP(env,Goals, next_state)
                    self.OPEN.put((new_node.f_value, new_node))
                    self.OPEN_SET[next_state] = new_node
                elif (next_state in self.OPEN_SET):
                    curr_node = self.OPEN_SET[next_state]
                    new_g_value = (node_to_expand.g_value + cost) # is this correct?
                    new_f_value = (1)*new_g_value + (h_weight)*self.h_MSAP(env, Goals, next_state)
                    if new_f_value < curr_node.f_value:
                        curr_node.g_value = new_g_value
                        curr_node.f_value = new_f_value
                        # python stores seperate versions on key and node.values so the order shouldn't be a problem
                        self.OPEN = updateKey(self.OPEN, self.OPEN_SET, curr_node, curr_node.f_value, new_f_value) #updates key in the queue
                else: # next_state is in CLOSED
                    curr_node = self.CLOSE[next_state]
                    new_g_value = (node_to_expand.g_value + cost) # is this correct?
                    new_f_value = (1)*new_g_value + (h_weight)*self.h_MSAP(env, Goals, next_state)
                    if new_f_value < curr_node.f_value:
                        curr_node.g_value = new_g_value
                        curr_node.f_value = new_f_value
                        self.OPEN.put((curr_node.f_value, curr_node))
                        self.OPEN_SET[next_state] = curr_node
                        del self.CLOSE[next_state]


class IDAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def search(self,  env: FrozenLakeEnv) -> Tuple[List[int],  int,  float]:
        raise NotImplementedError