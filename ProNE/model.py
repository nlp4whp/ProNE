# encoding=utf8
import os
import numpy as np
# import networkx as nx

import scipy.sparse
from scipy import linalg
from scipy.special import iv

from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

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
        1. Build model by:
            ```
            from ProNE import ProNE
            model = ProNE(graph_or_csr_file, node_number=None)
                :graph_or_csr_file: graph file, now we support three types of graph:
                        1. graph.txt    '1 2\n1 3\n2 3'
                        2. graph.txt    '1 2 3\n2 3'
                        3. graph.npz
                :node_number:       node count of graph, if None, model will get it by:
                        scan .txt file to get `max_node_num - min_node_num + 1`
                        get matrix0.shape[0] from .npz file

            # BTW: you can save it by:
                ```
                from ProNE import save_smat
                save_smat(csr_file, model.matrix0)
                ```
            ```
        2. Train model by:
            ```
            emb1, emb2 = model.train(dimension,
                                     step=10,
                                     theta=0.5,
                                     mu=0.2,
                                     emb_file1=None,
                                     emb_file2=None,
                                     smat_file=None)
                :dimension: embedding dimension
                :step:      iteration times
                :theta:     params of chebyshev_gaussian
                :mu:        params of chebyshev_gaussian
                :emb_file1: if None, emb1 (by RSVD) will not be saved
                :emb_file2: if None, emb2 (by chebyshev) will not be saved
                :smat_file: if None, CSR (model.matrix0) will not be saved

            # BTW:  you can also save emb1 and emb2 by:
                ```
                from ProNE import save_embedding
                save_embedding(emb_file1, emb1)
                save_embedding(emb_file2, emb1)
                ```
            ```
    Note:
        you can also load graph from CSR sparse matrix file, CSR file must has suffix `.npz`, like:
            ```
            model = ProNE('smat.npz', dimension, node_number=None)
            ```
    """

    def __init__(self, graph_or_csr_file, node_number=None):
        self.node_number = node_number  # if None, try scan `graph_file` to get `max_id + 1`
        """
        NOTE 加载二分图时, 构建networkx.Graph()会耗费大量内存, 因此完全改写为通过邻接图加载
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
        if graph_or_csr_file.endswith('.npz'):
            self._load_csr_mat(graph_or_csr_file)
        else:
            self._build_csr_mat(graph_or_csr_file)

    def _build_csr_mat(self, graph_file):
        """
        build adjacency by graph file, support both formats as:
            - '1 2\n1 3\n2 4\n2 7'
            - '1 2 3\n2 4 7'
        """
        print("| Initial graph and matrix0 ... |")
        if self.node_number is None:
            print(f"\t>> unknown `node_number`, scan {graph_file} to find max node_id")
            max_id = 1
            min_id = 1
            with open(graph_file) as fr:
                for line in fr:
                    max_id = max(max_id, *[int(i) for i in line.rstrip().split()])
                    min_id = min(min_id, *[int(i) for i in line.rstrip().split()])
                self.node_number = max_id - min_id + 1

        matrix0 = scipy.sparse.lil_matrix((self.node_number, self.node_number))
        with open(graph_file) as fr:
            for li, line in enumerate(fr):
                if li % 1000_0000 == 0:
                    print(f"- line: {li} ... ")
                try:
                    src, *tgt_list = map(int, line.rstrip().split())
                except Exception as e:
                    raise Exception(f"err line: {li}; {e}")
                for tgt in tgt_list:
                    if src != tgt:
                        matrix0[src, tgt] = 1
                        matrix0[tgt, src] = 1

        if matrix0.shape[0] != self.node_number:
            raise Exception(f"Unexcept node count: {matrix0.shape[0]}; it should be {self.node_number}")
        self.matrix0 = scipy.sparse.csr_matrix(matrix0)
        print(f"| Got matrix0.shape(`{self.matrix0.shape}`) ... |")

    def _load_csr_mat(self, smat_file):
        """
        just load CSR matrix file
        """
        self.matrix0 = scipy.sparse.load_npz(smat_file)
        self.node_number = self.matrix0.shape[0]

    def train(self, dimension, step=10, theta=0.5, mu=0.2,
              emb_file1=None, emb_file2=None, smat_file=None):
        """ ProNE training
        dimension: embedding dimension
        step: iteration times
        theta: params of chebyshev_gaussian
        mu: params of chebyshev_gaussian
        emb_file1: embedding_file1 got by factorization
        emb_file2: embedding_file2 got by chebyshev_gaussian
        """
        self.dimension = dimension
        with cost("| First factorization |"):
            features_matrix = self.pre_factorization(self.matrix0, self.matrix0)
            if isinstance(emb_file1, str):
                save_embedding(emb_file1, features_matrix)
        with cost("| Final chebyshev_gaussian; |"):
            embeddings_matrix = self.chebyshev_gaussian(
                                        A=self.matrix0,
                                        a=features_matrix,
                                        order=step,
                                        mu=mu,
                                        s=theta)
            if isinstance(emb_file2, str):
                save_embedding(emb_file2, embeddings_matrix)
        if isinstance(smat_file, str):
            save_smat(smat_file, self.matrix0)
        return features_matrix, embeddings_matrix

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

        A = scipy.sparse.eye(self.node_number) + A
        DA = preprocessing.normalize(A, norm='l1')
        L = scipy.sparse.eye(self.node_number) - DA

        M = L - mu * scipy.sparse.eye(self.node_number)

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
    _check_path(emb_file)
    with open(emb_file, 'w') as f_emb:
        f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
        for i in range(len(features)):
            s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
            f_emb.write(s + "\n")


def save_smat(smat_file, smat):
    """ save Adjacency into sparse matrix """
    _check_path(smat_file)
    try:
        scipy.sparse.save_npz(smat_file, smat)
    except Exception as e:
        print(Exception(f"Save CSR matrix0 ERROR: {e}"))


def _check_path(file_name):
    if file_name is None:
        return
    path, _ = os.path.split(file_name)
    if path == '':
        return
    if not os.path.isdir(path):
        os.makedirs(path)


# def run(graph_file, emb_file1, emb_file2, smat_file=None, node_num=None,
#         dim=100, step=10, theta=0.5, mu=0.2):

#     # check whether matrix file root has been created
#     _check_path(emb_file1)
#     _check_path(emb_file2)
#     _check_path(smat_file)

#     with cost("| Create ProNE model |"):
#         model = ProNE(graph_file, dim, node_number=node_num)
#         if smat_file is not None:
#             save_smat(smat_file, model.matrix0)

#     with cost("| First factorization |"):
#         features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
#         save_embedding(emb_file1, features_matrix)

#     with cost("| Final chebyshev_gaussian; |"):
#         embeddings_matrix = model.chebyshev_gaussian(
#                                         A=model.matrix0,
#                                         a=features_matrix,
#                                         order=step,
#                                         mu=mu,
#                                         s=theta)
#         save_embedding(emb_file2, embeddings_matrix)
