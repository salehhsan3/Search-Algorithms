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
        # self.h_value = 0
        # self.f_value = 0
        # self.weight = 0
    def __lt__(self, other):  # in case of duplicate values
        return self.state < other.state

class Agent:
    def __init__(self) -> None:
        self.OPEN_SET = dict()
        self.CLOSE = set()
        self.expanded_counter: int = 0
    def get_path(self, goal_node: GraphNode, expanded) -> Tuple[List[int] ,float, int]:
        lst = []
        node_iter = goal_node
        cost = 0
        while node_iter.parent is not None:
            lst.insert(0,node_iter.action)
            cost += node_iter.cost
            # advance the iterator:
            node_iter = node_iter.parent
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

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        initial_state = env.reset()
        initial_node = GraphNode(initial_state, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([],0,0)
        self.OPEN = queue.SimpleQueue()
        self.OPEN_SET = dict()
        self.CLOSE = set()
        self.expanded_counter = 0
        self.OPEN.put(initial_node)
        self.OPEN_SET[initial_node.state] = initial_node

        while not self.OPEN.empty():
            node_to_expand = self.OPEN.get()
            del self.OPEN_SET[node_to_expand.state]
            self.expanded_counter += 1
            self.CLOSE = self.CLOSE | {node_to_expand.state}

            for act,tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                if next_state is None and cost is None and terminated is None:
                    break  # it's a hole
                if terminated and env.is_final_state(next_state):
                    final_node = GraphNode(next_state, node_to_expand, act, cost)
                    return self.get_path(final_node, self.expanded_counter)
                if (next_state in self.OPEN_SET) or (next_state in self.CLOSE):
                    continue
                else:
                    new_node = GraphNode(next_state,node_to_expand,act,cost)
                    self.OPEN.put(new_node)
                    self.OPEN_SET[next_state] = new_node




class DFSAgent(Agent):
    def __init__(self) -> None:
        # dfs only needs a closed set to mark visited nodes
        super().__init__()

    # self.OPEN_SET = set()
    # self.CLOSE = set()
    # self.expanded_counter = 0.0

    def rec_dfs(self,graph_node,env: FrozenLakeEnv) -> GraphNode:
        if env.is_final_state(graph_node.state):
            return graph_node
        self.expanded_counter += 1
        self.CLOSE = self.CLOSE | {graph_node.state}

        for act, tup in env.succ(graph_node.state).items():
            next_state, cost, terminated = tup
            if next_state is None and cost is None and terminated is None:
                return None  # it's a hole
            if next_state in self.CLOSE:
                continue
            new_node = GraphNode(next_state,graph_node,act,cost)
            res = self.rec_dfs(new_node,env)
            if res is not None:
                return res
        return None




    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        initial_state = env.reset()
        initial_node = GraphNode(initial_state, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([],0,0)

        self.CLOSE = set()
        self.expanded_counter = 0

        goal_node = self.rec_dfs(initial_node,env)
        if goal_node is not None:
            return self.get_path(goal_node, self.expanded_counter)
        else:
            return [], 0, 0  # failure..


class UCSAgent(Agent):

    def __init__(self) -> None:
        # self.OPEN = queue.PriorityQueue()
        self.OPEN = heapdict.heapdict()
        super().__init__()

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        initial_state = env.reset()
        initial_node = GraphNode(initial_state, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([], 0, 0)
        self.OPEN = heapdict.heapdict()
        # self.OPEN_SET = dict()
        self.CLOSE = set()
        self.expanded_counter = 0
        self.OPEN[initial_state] = (initial_node.g_value, initial_node)
        # self.OPEN.put((initial_node.g_value,initial_node))
        # self.OPEN_SET[initial_node.state] = initial_node

        while len(self.OPEN) > 0:
            node_to_expand_idx, (g_value, node_to_expand) = self.OPEN.popitem()
            # del self.OPEN_SET[node_to_expand.state]
            self.CLOSE = self.CLOSE | {node_to_expand.state}
            if env.is_final_state(node_to_expand.state):
                return self.get_path(node_to_expand, self.expanded_counter)
            self.expanded_counter += 1

            for act, tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                if terminated and not env.is_final_state(next_state):
                    continue  # it's a hole, ignore it
                if (next_state not in self.OPEN) and (next_state not in self.CLOSE):
                     new_node = GraphNode(next_state, node_to_expand, act, cost)
                     new_node.g_value = node_to_expand.g_value + cost
                     self.OPEN[next_state] = (new_node.g_value, new_node)
                     # self.OPEN_SET[next_state] = new_node
                elif next_state in self.OPEN_SET:
                     next_node = self.OPEN_SET[next_state][1]
                     new_cost = node_to_expand.g_value + cost
                     if new_cost < next_node.g_value:
                         next_node.g_value = new_cost
                         next_node.parent = node_to_expand
                         next_node.action = act
                         next_node.cost = cost
                         self.OPEN[next_state] = (next_node.g_value, next_node)


class GreedyAgent(Agent):
  
    def __init__(self) -> None:
        self.OPEN = heapdict.heapdict()
        super().__init__()

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        initial_state = env.reset()
        initial_node = GraphNode(initial_state, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([], 0, 0)
        self.OPEN = heapdict.heapdict()
        # self.OPEN_SET = dict()
        self.CLOSE = set()
        self.expanded_counter = 0
        self.OPEN[initial_state] = (initial_node.g_value, initial_node)
        # self.OPEN_SET[initial_node.state] = initial_node

        while len(self.OPEN) > 0:
            node_to_expand_idx, (h_value, node_to_expand) = self.OPEN.popitem()
            # del self.OPEN_SET[node_to_expand.state]
            self.CLOSE = self.CLOSE | {node_to_expand.state}
            self.expanded_counter += 1

            for act, tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                if next_state is None and cost is None and terminated is None:
                    break  # it's a hole

                if (next_state not in self.OPEN) and (next_state not in self.CLOSE):
                    new_node: GraphNode = GraphNode(next_state, node_to_expand, act, cost)
                    if env.is_final_state(new_node.state):
                        return self.get_path(new_node, self.expanded_counter)
                    else:
                        self.OPEN[next_state] = (self.h_MSAP(env,env.goals,new_node.state), new_node)
                        # self.OPEN_SET[next_state] = new_node

class WeightedAStarAgent(Agent):
    
    def __init__(self):
        self.OPEN = heapdict.heapdict()
        self.CLOSED_DICT = {}
        super().__init__()

    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], float, int]:
        initial_state = env.reset()
        initial_node = GraphNode(initial_state, None, None, None)
        if env.is_final_state(initial_node.state):
            return ([], 0, 0)
        self.OPEN = heapdict.heapdict()
        # self.OPEN_SET = dict()
        self.CLOSE = set()
        self.CLOSED_DICT = {} # helpful in case a state is in CLOSED and we need to update it
        self.expanded_counter = 0
        self.OPEN[initial_state] = (initial_node.g_value, initial_node)

        while len(self.OPEN) > 0:
            node_to_expand_idx, (f_value, node_to_expand) = self.OPEN.popitem()
            self.CLOSE = self.CLOSE | {node_to_expand.state}
            self.CLOSED_DICT[node_to_expand.state] = node_to_expand
            if env.is_final_state(node_to_expand.state):
                return self.get_path(node_to_expand, self.expanded_counter)
            self.expanded_counter += 1

            for act, tup in env.succ(node_to_expand.state).items():
                next_state, cost, terminated = tup
                if terminated and not env.is_final_state(next_state):  # it's a hole: continue
                    continue
                elif (next_state not in self.OPEN) and (next_state not in self.CLOSE):
                    # hasn't been discovered yet,create a new graph node
                    new_node = GraphNode(next_state, node_to_expand, act, cost)
                    new_node.g_value = node_to_expand.g_value + cost

                    # w is between 0 and 1, thus we use the other version of wA*:
                    new_f_value = (1 - h_weight)*new_node.g_value + h_weight*self.h_MSAP(env, env.goals, next_state)

                    self.OPEN[next_state] = (new_f_value, new_node)
                elif next_state in self.OPEN:
                    next_node = self.OPEN[next_state][1]
                    new_cost = node_to_expand.g_value + cost
                    if new_cost < next_node.g_value: # since h_value and weight are consistent, the only difference is the g_value.
                        # shorter/cheaper path discovered, need to update g(v) and f(v) in queue:
                        next_node.g_value = new_cost
                        next_node.parent = node_to_expand
                        next_node.action = act
                        next_node.cost = cost
                        new_value = (1 - h_weight)*next_node.g_value + h_weight*self.h_MSAP(env, env.goals, next_state)
                        self.OPEN[next_state] = (new_value, next_node)
                else:
                    curr_node = self.CLOSED_DICT[next_state]
                    new_g_value = (node_to_expand.g_value + cost) 
                    new_f_value = (1-h_weight)*new_g_value + (h_weight)*self.h_MSAP(env, env.goals, next_state)
                    curr_f_value = (1-h_weight)*curr_node.g_value + (h_weight)*self.h_MSAP(env, env.goals, next_state)
                    if new_f_value < curr_f_value:
                        curr_node.g_value = new_g_value
                        curr_node.f_value = new_f_value
                        curr_node.action = act
                        curr_node.cost = cost
                        curr_node.parent = node_to_expand
                        self.OPEN[curr_node.state] = (new_f_value, curr_node)
                        self.CLOSE.remove(next_state)
                        del self.CLOSED_DICT[next_state]
                        # after this the node is in OPEN so we're not going to make a duplicate


class IDAStarAgent(Agent):
    def __init__(self):
        self.new_limit = 0.0
        self.current_path = []
        self.current_cost = 0.0
        super().__init__()

    def dfs_f(self, current_state, f_limit, env: FrozenLakeEnv) -> bool:
        f_value = self.current_cost + self.h_MSAP(env, env.goals, current_state)
        if f_value > f_limit:
            self.new_limit = min(self.new_limit, f_value)
            return False
        if env.is_final_state(current_state):
            return True

        self.expanded_counter += 1


        for act, tup in env.succ(current_state).items():
            next_state, cost, terminated = tup
            if (terminated and not env.is_final_state(next_state)) or (next_state == current_state):  # hole or visited
                continue

            self.current_path.append(act)
            self.current_cost += cost

            res = self.dfs_f(next_state, f_limit, env)
            if res is True:
                return res  # and don't touch current_path nor current_cost
            else:
                self.current_path.pop()
                self.current_cost -= cost

        return False  # Failure


    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], float, int]:
        self.new_limit = self.h_MSAP(env, env.goals, env.get_initial_state())
        self.expanded_counter = 0
        while True:
            initial_state = env.reset()
            f_limit = self.new_limit
            self.new_limit = np.inf
            self.current_path = []
            self.current_cost = 0.0
            # initial_node = GraphNode(initial_state, None, None, None)
            path = self.dfs_f(initial_state, f_limit, env)
            if path is True:
                return self.current_path, self.current_cost, self.expanded_counter

