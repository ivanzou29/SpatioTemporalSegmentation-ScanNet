# Copyright (c) Yunfan Zou (yunzou@student.ethz.ch). All Rights Reserved.
# github: ivanzou29

from collections import defaultdict


"""
    A CooccGraph object contains the information about the cooccurrences of instances of different classes
"""
class CooccGraph:
    def __init__(self, classes, coocc_graph):
        self.classes = classes
        self.graph = coocc_graph

    def getCooccClass(self, class_name):
        return self.graph[class_name]
    

"""
    A GlobalCooccGraph object that contains the global coocc graph, as well as the scene-level coocc graph for each scene
    Used to generate a loss function that is based on the cooccurrence of instances of different classes
"""
class GlobalCooccGraph:
    def __init__(self, classes, global_coocc_graph, coocc_graphs):
        self.classes = classes
        self.global_graph = global_coocc_graph
        self.graphs = coocc_graphs
        self.scene_weight_dict = self.init_scene_weights()

    def init_scene_weights(self):
        """
            For each edge, check its weight in the global graph, and make it inversely proportional to the weight in the global graph
        """

        scene_weight_dict = defaultdict(int)
        for scene_id in self.graphs:
            scene_weight = 0
            for i in range(len(self.classes) - 1):
                for j in range(i + 1, len(self.classes)):
                    scene_weight += self.get_coooc_graph_by_scene_id(scene_id)[classes[i]][classes[j]] * (1 / self.global_graph[classes[i]][classes[j]])
            
            scene_weight_dict[scene_id] = scene_weight
        
        return scene_weight_dict
    
    def get_coooc_graph_by_scene_id(self, scene_id):
        return self.graphs[scene_id]
    
    def get_scene_weight(self, scene_id):
        return self.scene_weight_dict[scene_id]
        