#!python3
import sys
import numpy as np
import time
import random
from math import log2
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def heuristic(candidates, curr, k, distance_func, data, level_, num_levels):
    if level_ < 2:
        threshold_angle = 0.2
    else:
        threshold_angle = min(1, 0.5 * (level_ ** 2) / num_levels)
    
    num_candidates = max(0, int((len(candidates)-1) * min(1,(level_/num_levels))))
    
    result = [candidates[0]]
    result_indx_set = {candidates[0][0]}
    added_data = [data[candidates[0][0]]]
    
    selected_candidates = random.sample(candidates[1:], num_candidates)
    not_selected = [candidate for candidate in candidates if candidate not in selected_candidates]
    
    for c_ in selected_candidates:
        result.append(c_)
        result_indx_set.add(c_[0])
        added_data.append(data[c_[0]])
    
    candidates = sorted(not_selected, key=lambda a: a[1])
    
    for c, curr_dist in candidates[1:]:
        c_data = data[c]
        angles = [np.dot(c_data, a) / (np.linalg.norm(c_data) * np.linalg.norm(a)) for a in added_data]
        min_angle = min(angles)
        if curr_dist < min([distance_func(c_data, a) for a in added_data]) and min_angle > threshold_angle:
            result.append((c, curr_dist))
            result_indx_set.add(c)
            added_data.append(c_data)
    
    for c, curr_dist in candidates:
        if len(result) < k and (c not in result_indx_set):
            result.append((c, curr_dist))
    
    return result

def k_closest(candidates: list, curr, k, distance_func, data):
    return sorted(candidates, key=lambda a: a[1])[:k]
    
class HNSW:
    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def __init__(self, distance_func, m=5, ef=10, ef_construction=50, m0=None, neighborhood_construction=heuristic, vectorized=False):
        self.data = []
        self.distance_func = distance_func
        self.neighborhood_construction = neighborhood_construction

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        self._m = m
        self._ef = ef
        self._ef_construction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 2 / log2(m)
        self._graphs = []
        self._enter_point = None
        
    def _find_new_median_point(self, layer):
        """Find a new median point based on the distances between elements."""
        node_indices = list(layer.keys())
        total_distances = {node: 0 for node in node_indices}
        for node, neighbors in layer.items():
            total_distances[node] = sum(dist for _, dist in neighbors)
        median_point = min(total_distances, key=total_distances.get)
        return median_point
        

    def add(self, elem, ef=None):
        if ef is None:
            ef = self._ef
        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        level = int(-log2(random.random()) * self._level_mult) + 1
        if level > len(graphs):
            level = len(graphs)+1
        idx = len(data)
        data.append(elem)

        if point is not None:
            for layer in reversed(graphs[level:]):
                point, dist = self.beam_search(graph=layer, q=elem, k=1, eps=[point], ef=1)[0]
            layer0 = graphs[0]
            num_of_layers = level
            total_num_layers = len(graphs)
            for j, layer in enumerate(reversed(graphs[:level])):
                level_m = m if layer is not layer0 else self._m0 # num of neighbors for layer
                candidates = self.beam_search(graph=layer, q=elem, k=level_m*2, eps=[point], ef=self._ef_construction)
                point = candidates[0][0]
                neighbors = self.neighborhood_construction(candidates=candidates, curr=idx, k=level_m, distance_func=self.distance_func, data=self.data,level_=num_of_layers, num_levels=total_num_layers)
                layer[idx] = neighbors
                for j, dist in neighbors:
                    candidates_j = layer[j] + [(idx, dist)]
                    neighbors_j = self.neighborhood_construction(candidates=candidates_j, curr=j, k=level_m, distance_func=self.distance_func, data=self.data, level_=num_of_layers, num_levels=total_num_layers)
                    layer[j] = neighbors_j
                num_of_layers -= 1
                           
        for i in range(len(graphs), level):
            graphs.append({idx: []})
        self._enter_point = self._find_new_median_point(graphs[-1])   
            
    # can be used for search after jump        
    def search(self, q, k=1, ef=10, level=0, return_observed=True):
        graphs = self._graphs
        point = self._enter_point
        for layer in reversed(graphs[level:]):
            point, dist = self.beam_search(layer, q=q, k=1, eps=[point], ef=1)[0]
            

        return self.beam_search(graph=graphs[level], q=q, k=k, eps=[point], ef=ef, return_observed=return_observed)

    def beam_search(self, graph, q, k, eps, ef, ax=None, marker_size=20, return_observed=False):
        max_ef = k
        '''
        graph – the layer where the search is performed
        q - query
        k - number of closest neighbors to return
        eps – entry points [vertex_id, ..., vertex_id]
        ef – size of the beam
        observed – if True returns the full of elements for which the distance were calculated
        returns – a list of tuples [(vertex_id, distance), ... , ]
        '''
        # Priority queue: (negative distance, vertex_id)
        

        candidates = []
        visited = set()  # set of vertex used for extending the set of candidates
        observed = dict() # dict: vertex_id -> float – set of vertexes for which the distance were calculated

        if ax:
            ax.scatter(x=q[0], y=q[1], s=marker_size, color='red', marker='^')
            ax.annotate('query', (q[0], q[1]))

        # Initialize the queue with the entry points
        for ep in eps:
            dist = self.distance_func(q, self.data[ep])
            heappush(candidates, (dist, ep))
            observed[ep] = dist

        while candidates:
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates)

            if ax:
                ax.scatter(x=self.data[current_vertex][0], y=self.data[current_vertex][1], s=marker_size, color='red')
                ax.annotate( len(visited), self.data[current_vertex] )

            observed_sorted = sorted( observed.items(), key=lambda a: a[1] )
            ef_largets = observed_sorted[ min(len(observed)-1, ef-1 ) ]
            if ef_largets[1] < dist:
                break
            #############################

            # Add current_vertex to visited set
            visited.add(current_vertex)

            # Check the neighbors of the current vertex
            for neighbor, _ in graph[current_vertex]:
                if neighbor not in observed:
                    dist = self.distance_func(q, self.data[neighbor])                    
                    # if neighbor not in visited:
                    heappush(candidates, (dist, neighbor))
                    observed[neighbor] = dist                    
                    if ax:
                        ax.scatter(x=self.data[neighbor][0], y=self.data[neighbor][1], s=marker_size, color='yellow')
                        # ax.annotate(len(visited), (self.data[neighbor][0], self.data[neighbor][1]))
                        ax.annotate(len(visited), self.data[neighbor])
                    
        observed_sorted =sorted( observed.items(), key=lambda a: a[1] )
        if return_observed:
            return observed_sorted
        return observed_sorted[:k]
    
    def save_graph_plane(self, file_path):
        with open(file_path, "w") as f:
            f.write(f'{len(self.data)}\n')

            for x in self.data:
                s = ' '.join([a.astype('str') for a in x ])
                f.write(f'{s}\n')

            for graph in self._graphs:
                for src, neighborhood in graph.items():
                    for dst, dist in neighborhood: 
                        f.write(f'{src} {dst}\n')