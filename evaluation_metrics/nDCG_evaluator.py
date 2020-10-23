import numpy as np
from sklearn.metrics import ndcg_score


class nDCG_Evaluator():

    def __init__(self, node_json_data, node_topic_json_data):
        self.node_json_data = node_json_data
        self.node_topic_json_data = node_topic_json_data
        self.node_id_dict = {}
        self.node_name_dict = {}
        for node_id in self.node_json_data.keys():
            self.node_id_dict[node_id] = self.node_json_data[node_id]
            self.node_name_dict[self.node_json_data[node_id]] = node_id

    def calculate_nDCG_score(self, source_node_name, target_node_names):
        nDCG_score = 0
        if source_node_name in self.node_name_dict.keys():
            rank_scores = []
            for target_node_name in target_node_names:
                if target_node_name in self.node_name_dict.keys():
                    source_node_id = self.node_name_dict[source_node_name]
                    target_node_id = self.node_name_dict[target_node_name]
                    rank_score = self.__rank_query(source_node_id, target_node_id)
                    rank_scores.append(rank_score)

            ground_truths = np.asarray([[3 for i in rank_scores]])
            rank_scores = np.asarray([rank_scores])
            nDCG_score = ndcg_score(rank_scores, ground_truths)
        return nDCG_score

    def __rank_query(self, src_node_id, target_node_id):
        rank_score = 0
        src_topic_set = self.node_topic_json_data[src_node_id]
        target_topic_set = self.node_topic_json_data[target_node_id]
        denominator = len(src_topic_set)
        if len(target_topic_set) < len(target_topic_set):
            denominator = len(target_topic_set)
        topic_matching_percentage = (len(self.__intersection(src_topic_set, target_topic_set)) / denominator) * 100
        if topic_matching_percentage >= 20 and topic_matching_percentage <= 39:
            rank_score = 1
        if topic_matching_percentage >= 40 and topic_matching_percentage <= 59:
            rank_score = 2
        if topic_matching_percentage >= 60:
            rank_score = 3
        return rank_score

    def __intersection(self, a, b):
        return list(set(a) & set(b))
