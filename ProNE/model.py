# encoding=utf8
import os
import numpy as np
# import networkx as nx

import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv

from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

import argparse
import time
from contextlib import contextmanager


@contextmanager
def cost(note=''):
    print(f"{note} Start")
    start = time.time()
    yield
    print(f"{note} OVER => cost: {time.time() - start:.4f}sec")


class ProNE():

    """
    Tutorial:
        1. build model by:
            ```
            from ProNE.model import ProNE
            from ProNE.model import save_smat
            from ProNE.model import save_embedding

            model = ProNE(graph_file, dimension, node_number=None)
                :graph_file:   bi-graph file, saved adjacency like: 'node_i node_j'
                :dimension:    embedding dimension
                :node_number:  node count of graph, if None, model need scan `graph_file` to find it
            save_smat(smat_fn, model.model0)
            ```
        2. Embedding by factorization
            ```
            features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
            save_embedding(emb_fn1, features_matrix)
            ```
        3. Embedding by chebyshev_gaussian
            ```
            embeddings_matrix = model.chebyshev_gaussian(
                                        A=model.matrix0,
                                        a=features_matrix,
                                        order=step,
                                        mu=mu,
                                        s=theta)
            save_embedding(emb_fn2, embeddings_matrix)
            ```
    Also:
        you can run all steps together by:
            ```
            from ProNE.model import run
            run(graph_file, emb_fn1, emb_fn2, smat_fn, node_num=None,
                dim=100, step=10, theta=0.5, mu=0.2)
            ```
    """

    def __init__(self, graph_file, dimension, node_number=None):
        self.graph = graph_file
        self.dimension = dimension
        self.node_number = node_number  # if None, try scan `graph_file` to get `max_id + 1`

        """
        NOTE 这部分需要加载二分图, 构建networkx.Graph(), 会耗费大量内存
        self.G = nx.read_edgelist(self.graph, nodetype=int, create_using=nx.DiGraph())
        self.G = self.G.to_undirected()
        self.node_number = self.G.number_of_nodes()
        matrix0 = scipy.sparse.lil_matrix((self.node_number, self.node_number))

        for e in self.G.edges():
            if e[0] != e[1]:
                matrix0[e[0], e[1]] = 1
                matrix0[e[1], e[0]] = 1
        self.matrix0 = scipy.sparse.csr_matrix(matrix0)
        print(matrix0.shape)
        """

        # Build Adjacency matrix0 by bi-graph `graph_file`
        print(f"| Initial graph and matrix0 ... |")
        if self.node_number is None:
            print(f"\t>> unknown `node_number`, scan {graph_file} to find max node_id")
            max_id = -1
            with open(graph_file) as fr:
                for line in fr:
                    max_id = max(max_id, *[int(i) for i in line.rstrip().split()])
                self.node_number = max_id + 1
        matrix0 = scipy.sparse.lil_matrix((self.node_number, self.node_number))
        with open(graph_file) as fr:
            for li, line in enumerate(fr):
                if li % 1000_0000 == 0:
                    print(f"- line: {li} ... ")
                try:
                    src, tgt = map(int, line.rstrip().split())
                except Exception as e:
                    raise Exception(f"err line: {li}; {e}")
                if src != tgt:
                    matrix0[src, tgt] = 1
                    matrix0[tgt, src] = 1

        if matrix0.shape[0] != self.node_number:
            raise Exception(f"Unexcept node count: {matrix0.shape[0]}; it should be {self.node_number}")
        self.matrix0 = scipy.sparse.csr_matrix(matrix0)
        print(f"| Got matrix0.shape({self.matrix0.shape}) ... |")

    def get_embedding_rand(self, matrix):
        """ Sparse randomized tSVD for fast embedding """
        ln = matrix.shape[0]
        smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
        print('svd sparse', smat.data.shape[0] * 1.0 / ln ** 2)
        U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        return U

    def get_embedding_dense(self, matrix, dimension):
        """ get dense embedding via SVD """
        U, s, Vh = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
        U = np.array(U)
        U = U[:, :dimension]
        s = s[:dimension]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        return U

    def pre_factorization(self, tran, mask):
        """ Network Embedding as Sparse Matrix Factorization """
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / neg.sum()

        neg = scipy.sparse.diags(neg, format="csr")
        neg = mask.dot(neg)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1
        features_matrix = self.get_embedding_rand(F)
        return features_matrix

    def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        """ NE Enhancement via Spectral Propagation """
        print('Chebyshev Series -----------------')

        if order == 1:
            return a

        A = sp.eye(self.node_number) + A
        DA = preprocessing.normalize(A, norm='l1')
        L = sp.eye(self.node_number) - DA

        M = L - mu * sp.eye(self.node_number)

        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a

        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            #         Lx2 = 2*L.dot(Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
        mm = A.dot(a - conv)
        emb = self.get_embedding_dense(mm, self.dimension)
        return emb


def save_embedding(emb_file, features):
    """ save node embedding into `emb_file` with word2vec format """
    with open(emb_file, 'w') as f_emb:
        f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
        for i in range(len(features)):
            s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
            f_emb.write(s + "\n")


def save_smat(smat_fn, smat):
    """ save Adjacency into sparse matrix """
    scipy.sparse.save_npz(smat_fn, smat)


def run(graph_file, emb_fn1, emb_fn2, smat_fn, node_num=None,
        dim=100, step=10, theta=0.5, mu=0.2):
    """
    graph_file: bi-graph file, saved adjacency like: 'node_i node_j\nnode_i node_k'
    emb_fn1: embedding_file1 got by factorization
    emb_fn2: embedding_file2 got by chebyshev_gaussian
    smat_fn: sparse matrix file
    dim: ProNE embedding dimension
    step: iteration times
    theta: params of chebyshev_gaussian
    mu: params of chebyshev_gaussian
    """

    # check whether matrix file root has been created
    if not os.path.isdir(os.path.split(emb_fn1)[0]):
        os.makedirs(os.path.split(emb_fn1)[0])
    if not os.path.isdir(os.path.split(emb_fn2)[0]):
        os.makedirs(os.path.split(emb_fn2)[0])
    if not os.path.isdir(os.path.split(smat_fn)[0]):
        os.makedirs(os.path.split(smat_fn)[0])

    with cost(f"| Create ProNE model |"):
        model = ProNE(graph_file, dim, node_number=node_num)
        save_smat(smat_fn, model.matrix0)

    with cost(f"| First factorization |"):
        features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
        save_embedding(emb_fn1, features_matrix)

    with cost(f"| Final chebyshev_gaussian; |"):
        embeddings_matrix = model.chebyshev_gaussian(
                                        A=model.matrix0,
                                        a=features_matrix,
                                        order=step,
                                        mu=mu,
                                        s=theta)
        save_embedding(emb_fn2, embeddings_matrix)
