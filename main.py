import argparse
import time
import timeout_decorator
import pickle

import chainer
import chainer.functions as F
import networkx as nx
import dwave_networkx as dnx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import seaborn as sns

from uf import UnionFind

np.random.seed(1124)


def int_set(subset):
    """
    set to corresponding integer representation
    """
    rep = 0
    for i in subset:
        rep += (1 << i)

    return rep


def set_str(subset, v):
    """
    This function transforms a subset and a vertex into corresponding string representation.
    """
    return str(int_set(subset)) + " " + str(v)


def Q(G, S, v):
    """
    This function calculates Q(S, v) = {w ∈ V − S − {v} | there is a path from v to w in G[S ∪ {v, w}]}.
    The time complexity of this function is O(n + m) time.

    Parameters
    ----------
    G : Graph object
        entire graph
    S : set
        a subset of vertices
    v : int
        vertex

    Returns
    -------
    {w ∈ V − S − {v} | there is a path from v to w in G[S ∪ {v, w}]} : int
        The string representation of a given subset and a given vertex.
    """
    cnt = 0
    induce_V = S
    induce_V.add(v)

    # initialize union find tree
    uf = UnionFind(len(G.nodes))

    # calc connected component
    for edge in G.edges:
        (source, sink) = edge
        if source in induce_V and sink in induce_V:
            uf.Unite(source, sink)

    # enum paths
    for vertex in (set(G.nodes) - S - set([v])):
        # v and w is same connected component?
        for neighbor in G[vertex]:
            if uf.isSameGroup(v, neighbor):
                cnt += 1
                break

    return cnt


class TreewidthAlgorithm():
    """
    A class for algorithms calculates treewidth.
    The attributes in this class have some

    Attributes
    ----------
    dp_S : set
        memorize the result of TWCheck(G, S)
    dp_Q : set
        memorize the result of Q(G, S, v)
    bound : string
        use upper-prune or lower-prune or both
    prune_num : int
        memorize the number of pruned search state
    func_call_num : int
        memorize the number of function calls
    eval_GNN_time : float
        memorize the time of evaluation GNN
    """
    def __init__(self, prune):
        self.dp_S = {}
        self.dp_Q = {}
        self.prune = prune
        self.func_call_num = 0

    def initialize(self):
        # O(|dp_S|)
        if self.prune == "upper":
            # Upper-Calculation -> preserve false
            new_S = {}
            for k, v in self.dp_S.items():
                if not v:
                    new_S[k] = v
            self.dp_S = new_S

        elif self.prune == "lower":
            # Lower-Calculation -> preserve true
            new_S = {}
            for k, v in self.dp_S.items():
                if v:
                    new_S[k] = v
            self.dp_S = new_S

    def Q(self, G, S, v):
        '''
        This method calculates Q(S, v) = {w ∈ V − S − {v} | there is a path from v to w in G[S ∪ {v, w}]}.
        The time complexity of this function is O(n + m) time. This method uses dp_Q.

        Parameters
        ----------
        G : Graph object
            entire graph
        S : set
            a subset of vertices
        v : int
            vertex

        Returns
        -------
        {w ∈ V − S − {v} | there is a path from v to w in G[S ∪ {v, w}]} : int
            The string representation of a given subset and a given vertex.
        '''
        if set_str(S, v) in self.dp_Q:
            return self.dp_Q[set_str(S, v)]

        cnt = 0
        induce_V = S
        induce_V.add(v)

        # initialize union find tree
        uf = UnionFind(len(G.nodes))

        # calc connected component
        for edge in G.edges:
            (source, sink) = edge
            if source in induce_V and sink in induce_V:
                uf.Unite(source, sink)

        # enum paths
        for vertex in (set(G.nodes) - S - set([v])):
            # v and w is same connected component?
            for neighbor in G[vertex]:
                if uf.isSameGroup(v, neighbor):
                    cnt += 1
                    break

        self.dp_Q[set_str(S, v)] = cnt
        return cnt

    def calc_treewidth_recursive(self, G, S, opt):
        """
        This method judges whether G[S] has treewidth at most opt by using recursive algorithm.
        This method uses dp_S to reduce computational complexity.

        Parameters
        ----------
        G : Graph object
            entire graph
        S : set
            a subset of vertices
        opt : int
            a value to judge

        Returns
        -------
        Result : boolean
            whether the given graph has treewidth at most opt
        """
        self.func_call_num += 1

        if len(S) == 1:
            return self.Q(G, set(), S.pop()) <= opt

        if int_set(S) in self.dp_S:
            return self.dp_S[int_set(S)]

        res = False

        for vertex in S:
            Qval_check = self.Q(G, S - set([vertex]), vertex) <= opt
            if Qval_check:
                res = (res or self.calc_treewidth_recursive(G, S - set([vertex]), opt))
            if res:
                break

        self.dp_S[int_set(S)] = res
        return res

def ordinaryDP(eval_graph, calc_type):
    calc_DP_tw = TreewidthAlgorithm(calc_type)

    check_range = range(1, eval_graph.number_of_nodes()) if calc_type == "lower" else range(eval_graph.number_of_nodes() - 1, -1, -1)

    start = time.time()
    for opt in check_range:
        DP_tw = calc_DP_tw.calc_treewidth_recursive(eval_graph, eval_graph.nodes, opt)
        if (DP_tw and (calc_type == "lower")) or ((not DP_tw) and (calc_type == "upper")):
            break
        calc_DP_tw.initialize()
    end = time.time()

    return end - start, opt + (0 if calc_type == "lower" else 1), calc_DP_tw.func_call_num


def main(args):
    print('--- Prediction by existing algorithm ---')
    print('Index\t|V|\t|E|\ttw(G)\ttime\tevaltw\tprunenum\tfunccallnum')
    output_file = "exist.dat"
    result = ["Index\t|V|\t|E|\ttw\ttime\tevaltw\tfunccallnum"]

    # make some graphs
    graphs = []
    for idx in range(0, args.data_num):
        n = np.random.randint(5, 15) # [5, 10] |V|
        m = np.random.randint(n - 1, (n * (n - 1)) // 2) # |E|
        g = nx.dense_gnm_random_graph(n, m)
        while not nx.is_connected(g):
            g = nx.dense_gnm_random_graph(n, m)

        graphs.append(g)


    for idx, eval_graph in enumerate(graphs):
        # exact computation
        tw = dnx.treewidth_branch_and_bound(eval_graph)[0]

        for calc_type in ["upper", "lower"]:
            graphstat = "{3}\t{0}\t{1}\t{2}".format(eval_graph.number_of_nodes(), eval_graph.number_of_edges(), tw, str(idx).rjust(5))
            print('{0}\t{1}\t{2}\t{3}'.format(str(idx).rjust(5), eval_graph.number_of_nodes(), eval_graph.number_of_edges(), tw), end='\t')
            try:
                tm, evtw, fcn = ordinaryDP(eval_graph, calc_type)
                res = "{0}\t{1}\t{2}".format(tm, evtw, fcn)
            except TimeoutError:
                res = "TimeOut"
            print(res)
            assert evtw == tw
            result.append(graphstat + "\t" + res)

    # write results to a file
    if args.out_to_file:
        with open("./{}/Approach1/".format(args.out) + output_file, "w") as f:
            f.write('\n'.join(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A implementation of treewidth calculation algorithms')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--out_to_file', action='store_true',
                        help='output the results')
    parser.add_argument('--data_num', type=int, default=100,
                        help='the number of graphs you want to calculate treewidth')
    args = parser.parse_args()

    main(args)
