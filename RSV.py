import math
import copy
class RSV:
    def __init__(self, N, E, background, foreground, coalitions): 
        self.N = set(N) - set(max(N))
        self.N.add(0) 
        self.E = [set() for _ in range(len(N) + 1)]
        for edge in E: self.E[edge[0]].add(edge)
        children = set()
        for edge in E: children.add(edge[1])
        sources = N - children
        for source in sources: self.E[0].add((0, source))
        self.E.pop()
        self.background = background
        self.foreground = foreground
        self.coalitions = coalitions

    def _parents(self, child):
        parent = set()
        for node in self.N:
            for edge in self.E[node]:
                if edge[1] == child:
                    parent.add(edge[0])
        return parent
    
    def _RSV_helper(self, edge, curr_edges):
        π = 0
        coalition = self.coalitions[edge[0]]
        for sub_active in coalition:
            if edge not in sub_active:
                temp_edges = copy.deepcopy(curr_edges)
                temp_edges[edge[0]] = sub_active
                if edge[0] > 0:
                    π += self._shapley_value(sub_active, edge[0]) * sum([self._RSV_helper((parent, edge[0]), curr_edges) - self._RSV_helper((parent, edge[0]), temp_edges) for parent in self._parents(edge[0])])
                else:
                    π += self._shapley_value(sub_active, edge[0]) * (self._value(curr_edges) - self._value(temp_edges))
        return π

    def _value(self, final_edges):
        X = [(i, 'b') for i in self.background]
        for node in self.N:
            for edge in final_edges[node]:
                    if edge[0] == 0 or X[edge[0] - 1][1] == 'f':
                        X[edge[1] - 1] = (self.foreground[edge[1] - 1], 'f')
        return X[-1][0]

    def _shapley_value(self, sub_active, node):
        return math.factorial(len(sub_active)) * math.factorial(len(self.E[node]) - len(sub_active) - 1) / math.factorial(len(self.E[node]))

    def RSV(self):
        attributions = dict()
        for node in self.N:
            for edge in self.E[node]: 
                attributions[edge] = self._RSV_helper(edge, copy.deepcopy(self.E))
        return attributions

if __name__ == "__main__":
    model = RSV({1,2,3}, set([(1, 2), (2, 3)]), [0, 0, 0], [1, 1, 1], [[{}, {(0, 1)}], [{}, {(1, 2)}], [{}, {(2, 3)}]])
    print(model.RSV())