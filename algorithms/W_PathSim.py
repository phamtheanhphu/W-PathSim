import operator
import numpy as np
from utils.C_HIN import C_HIN


class W_PathSim():

    def __init__(self, input_graph_file, metapath):

        # Reading the content-based heterogeneous information network (C-HIN)
        self.input_graph_file = input_graph_file
        self.C_HIN = C_HIN(self.input_graph_file)

        self.metapath = metapath

        # Analyzing given C-HIN with the defined metapath
        self.__preprocess_network()

    def __preprocess_network(self):
        self.C_HIN.analyze_with_metapath(self.metapath)

    def __get_path_instances_of_pairwise_nodes(self, src_node_id, dst_node_id):
        path_instances_of_pairwise_nodes = []
        if len(self.C_HIN.path_instances) > 0:
            for path_instance in self.C_HIN.path_instances:
                if path_instance[0] == src_node_id and path_instance[-1] == dst_node_id:
                    path_instances_of_pairwise_nodes.append(path_instance)
        return path_instances_of_pairwise_nodes

    def __w_pathsim_score(self, src_node_id, dst_node_id):

        # Calculate: W-PathSim(src -> dst)
        src2dst_path_instances = self.__get_path_instances_of_pairwise_nodes(src_node_id, dst_node_id)
        src2dst_weight = 1
        if self.C_HIN.is_content_based_metapath():
            src2dst_path_instance_weights = [self.C_HIN.calc_path_instance_weight(path_instance)
                                             for path_instance in src2dst_path_instances]
            if len(src2dst_path_instance_weights) > 0:
                src2dst_weight = np.mean(src2dst_path_instance_weights)

        # Calculate: W-PathSim(src -> src)
        src2src_path_instances = self.__get_path_instances_of_pairwise_nodes(src_node_id, src_node_id)
        src2src_weight = 1
        if self.C_HIN.is_content_based_metapath():
            src2src_path_instance_weights = [self.C_HIN.calc_path_instance_weight(path_instance)
                                             for path_instance in src2src_path_instances]
            if len(src2src_path_instance_weights) > 0:
                src2src_weight = np.mean(src2src_path_instance_weights)

        # Calculate: W-PathSim(dst -> dst)
        dst2dst_path_instances = self.__get_path_instances_of_pairwise_nodes(dst_node_id, dst_node_id)
        dst2dst_weight = 1
        if self.C_HIN.is_content_based_metapath():
            dst2dst_path_instance_weights = [self.C_HIN.calc_path_instance_weight(path_instance)
                                             for path_instance in dst2dst_path_instances]
            if len(dst2dst_path_instance_weights) > 0:
                dst2dst_weight = np.mean(dst2dst_path_instance_weights)

        sim_score = 0
        numerator = src2dst_weight * len(src2dst_path_instances)
        denominator = (src2src_weight * len(src2src_path_instances)) + (dst2dst_weight * len(dst2dst_path_instances))
        if denominator > 0:
            sim_score = 2 * (numerator / denominator)
        return sim_score

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_top_k_similar_nodes(self, src_node_id, k):
        normalized_sim_scores_dict = {}
        unnormalized_sim_scores = []
        for (dst_node_id, node_attrs) in self.C_HIN.nodes:
            if self.C_HIN.nodes[src_node_id][self.C_HIN.NODE_TYPE_ATTR] == node_attrs[self.C_HIN.NODE_TYPE_ATTR]:
                if src_node_id != dst_node_id:
                    src2dst_w_pathsim_score = self.__w_pathsim_score(src_node_id, dst_node_id)
                    normalized_sim_scores_dict[dst_node_id] = src2dst_w_pathsim_score
                    unnormalized_sim_scores.append(src2dst_w_pathsim_score)

        # normalizing the W-PathSim similarity score -> range [0, 1]
        normalized_sim_scores = self.__sigmoid(np.array(unnormalized_sim_scores))
        for idx, node_id in enumerate(normalized_sim_scores_dict.keys()):
            normalized_sim_scores_dict[node_id] = normalized_sim_scores[idx]

        sorted_normalized_sim_scores = sorted(normalized_sim_scores_dict.items(), key=operator.itemgetter(1),
                                              reverse=True)
        if len(sorted_normalized_sim_scores) > k:
            top_k_results = sorted_normalized_sim_scores[:k]
        else:
            top_k_results = sorted_normalized_sim_scores

        return top_k_results
