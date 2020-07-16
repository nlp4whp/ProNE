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


class ProNE():
    def __init__(self, graph_file, emb_file1, emb_file2, dimension, node_number=None):
        self.graph = graph_file
        self.emb1 = emb_file1
        self.emb2 = emb_file2
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

        if not os.path.isdir(os.path.split(self.emb_file1)[0]):
            os.makedirs(os.path.split(self.emb_file1)[0])
        if not os.path.isdir(os.path.split(self.emb_file2)[0]):
            os.makedirs(os.path.split(self.emb_file2)[0])

        # 根据二分图构建邻接矩阵matrix0
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
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        ln = matrix.shape[0]
        smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
        print('svd sparse', smat.data.shape[0] * 1.0 / ln ** 2)
        U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        print('sparsesvd time', time.time() - t1)
        return U

    def get_embedding_dense(self, matrix, dimension):
        # get dense embedding via SVD
        t1 = time.time()
        U, s, Vh = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
        U = np.array(U)
        U = U[:, :dimension]
        s = s[:dimension]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        print('densesvd time', time.time() - t1)
        return U

    def pre_factorization(self, tran, mask):
        # Network Embedding as Sparse Matrix Factorization
        t1 = time.time()
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / neg.sum()

        neg = scipy.sparse.diags(neg, format="csr")
        neg = mask.dot(neg)
        print("neg", time.time() - t1)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1
        features_matrix = self.get_embedding_rand(F)
        return features_matrix

    def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        # NE Enhancement via Spectral Propagation
        print('Chebyshev Series -----------------')
        t1 = time.time()

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
            print('Bessell time', i, time.time() - t1)
        mm = A.dot(a - conv)
        emb = self.get_embedding_dense(mm, self.dimension)
        return emb


def save_embedding(emb_file, features):
    # save node embedding into emb_file with word2vec format
    f_emb = open(emb_file, 'w')
    f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
    for i in range(len(features)):
        s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
        f_emb.write(s + "\n")
    f_emb.close()


def save_smat(smat_file, smat):
    scipy.sparse.save_npz(smat_file, smat)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ProNE.")
    parser.add_argument('-graph', nargs='?', default='data/blogcatalog.ungraph',
                        help='Graph path')
    parser.add_argument('-emb1', nargs='?', default='emb/blogcatalog.emb',
                        help='Output path of sparse embeddings')
    parser.add_argument('-emb2', nargs='?', default='emb/blogcatalog_enhanced.emb',
                        help='Output path of enhanced embeddings')
    parser.add_argument('-smat', default='emb/smat.npz',
                        help='Output path of sparse matrix')
    parser.add_argument('-dimension', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('-node_num', type=int, default=None,
                        help='Number of nodes. `Max_node_id + 1`')
    parser.add_argument('-step', type=int, default=10,
                        help='Step of recursion. Default is 10.')
    parser.add_argument('-theta', type=float, default=0.5,
                        help='Parameter of ProNE. Default is 0.5.')
    parser.add_argument('-mu', type=float, default=0.2,
                        help='Parameter of ProNE. Default is 0.2')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"| Create ProNE model |")
    t_0 = time.time()
    model = ProNE(args.graph, args.emb1, args.emb2, args.dimension, node_number=args.node_num)
    t_1 = time.time()
    print(f"| Create ProNE model; cost {t_1 - t_0} |")
    save_smat(smat_file=args.smat, smat=model.matrix0)

    print(f"| First factorization |")
    features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
    t_2 = time.time()
    print(f"| First factorization; cost {t_2 - t_1} |")
    save_embedding(args.emb1, features_matrix)

    print(f"| Final chebyshev_gaussian; |")
    embeddings_matrix = model.chebyshev_gaussian(model.matrix0, features_matrix, args.step, args.mu, args.theta)
    t_3 = time.time()
    print(f"| Final chebyshev_gaussian; cost {t_3 - t_2} |")
    save_embedding(args.emb2, embeddings_matrix)

    print('---', model.node_number)
    print('total time', t_3 - t_0)
    print('sparse NE time', t_2 - t_1)
    print('spectral Pro time', t_3 - t_2)


if __name__ == '__main__':
    """
    python proNE.py \
    -graph data/line.adj.ungraph \
    -emb1 emb/line_sparse.adj.emb \
    -emb2 emb/line_spectral.adj.emb \
    -dimension 128 \
    -step 10 \
    -theta 0.5 \
    -mu 0.2
    """
    main()
