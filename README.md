# ProNE

### [Paper](https://www.ijcai.org/proceedings/2019/594)

ProNE: Fast and Scalable Network Representation Learning

Jie Zhang, [Yuxiao Dong](https://ericdongyx.github.io/), Yan Wang, [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/) and Ming Ding

Accepted to IJCAI 2019 Research Track!

## Prerequisites

- Linux or macOS
- Python 2 or 3
- scipy
- sklearn


## Installation

Clone this repo.

```bash
git clone https://github.com/lykeven/ProNE
cd ProNE
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

## Dataset

These datasets are public datasets.

- PPI contains 3,890 nodes, 76,584 edges and 60 labels.
- Wikipedia contains 4,777 nodes, 184,812 edges and 40 labels.
- Blogcatalog contains 10,312 nodes, 333,983 edges and 39 labels.
- DBLP contains 51,264 nodes, 127,968 edges and 60 labels. 
- Youtube contains 1,138,499 nodes, 2,990,443 edges and 47 labels.

## Training

### Training on the existing datasets

Create emb directory to save output embedding file
```bash
mkdir emb
```
You can use `python proNE.py -graph example_graph` to train ProNE model on the example data.

If you want to train on the PPI dataset, you can run 

```bash
python proNE.py -graph data/PPI.ungraph -emb1 emb/PPI_sparse.emb -emb2 emb/PPI_spectral.emb
 -dimension 128 -step 10 -theta 0.5 -mu 0.2
```
Where PPI_sparse.emb and PPI_spectral.emb are output embedding files and dimension=128, step=10, theta=0.5 and mu=0.2 are the default setting for a good result. Better results would be achieved when searching mu over values around 0, for example, the results when mu = -4.0 (low pass) on Wikipedia in the enhancement experiments are better than those reported in the paper.
If you want to evaluate the embedding via node classification task, you can run

```bash
python classifier.py -label data/PPI.cmty -emb emb/PPI_spectral.emb -shuffle 4
```
Where PPI.cmty are node label file and shuffle is the number of shuffle times for classification.

### Training on your own datasets

If you want to train ProNE on your own dataset, you should prepare the following files:
- edgelist.txt: Each line represents an edge, which contains two tokens `<node1> <node2>` where each token is a number starting from 0.

### Training on c++ version ProNE
ProNE is mainly single-thread(except for the svd on small matrices). We also provide a c++ multi-thread program ProNE.cpp for large-scale network based on
 [Eigen](http://eigen.tuxfamily.org), [MKL](https://software.intel.com/en-us/mkl), [FrPCA](https://github.com/XuFengthucs/frPCA_sparse/) and [boost](https://www.boost.org/). [Openmp](https://www.openmp.org/), and [ICC](https://software.intel.com/en-us/c-compilers) are used to speed up. Besides, [gflags](https://github.com/gflags/gflags) is required to parse command parameter.

Compared with the orginal python version ProNE in the paper, C++ ProNE under all optimization is about 6 times faster (two minutes)  on youtube without the loss of acurracy performance.

Compile it via
```bash
icc ProNE.cpp -O3 -mkl -qopenmp -l gflags frpca/frpca.c frpca/matrix_vector_functions_intel_mkl_ext.c frpca/matrix_vector_functions_intel_mkl.c  -o ProNE.out
```

If you want to train on the PPI dataset, you can run
```bash
./ProNE.out -filename data/PPI.ungraph -emb1 emb/PPI.emb1 -emb2 emb/PPI.emb2
 -num_node 3890 -num_step 10 -num_thread 20 -num_rank 128 -theta 0.5 -mu 0.2
```


If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.


## Citing

If you find *ProNE* is useful for your research, please consider citing our paper:

```
@inproceedings{ijcai2019-594,
  title     = {ProNE: Fast and Scalable Network Representation Learning},
  author    = {Zhang, Jie and Dong, Yuxiao and Wang, Yan and Tang, Jie and Ding, Ming},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {4278--4284},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/594},
  url       = {https://doi.org/10.24963/ijcai.2019/594},
}
```

## Build, Upload, Install

``` shell
## build whl
python setup.py bdist_wheel
# or
./build.sh

## upload
twine upload *.whl -r {pypirc-index-server}
# or
twine upload *.whl --username {htpasswd-username} --repository-url {pypirc-index-server-repository-url}

## install
pip install *.whl
# or
pip install ProNE -i http://{ip}:{port} --trusted-host {ip}

## test if sucessed
python -c "from ProNE import ProNE;help(ProNE)"
cd test
python test_prone.py
```

## Usage
  - Tutorial:

    1. Build model by:
        ``` python
        from ProNE import ProNE
        model = ProNE(graph_or_csr_file, node_number=None)
            """
            :graph_or_csr_file: graph file, now we support three types of graph:
                    1. graph.txt    '1 2\n1 3\n2 3'
                    2. graph.txt    '1 2 3\n2 3'
                    3. graph.npz
            :node_number:       node count of graph, if None, model will get it by:
                    scan .txt file to get `max_node_num - min_node_num + 1`
                    get matrix0.shape[0] from .npz file
            """
        # BTW: you can save it by:
        from ProNE import save_smat
        save_smat(csr_file, model.matrix0)
        ```
    2. Train model by:
        ``` python
        emb1, emb2 = model.train(dimension,
                                 step=10,
                                 theta=0.5,
                                 mu=0.2,
                                 emb_file1=None,
                                 emb_file2=None,
                                 smat_file=None)
            """
            :dimension: embedding dimension
            :step:      iteration times
            :theta:     params of chebyshev_gaussian
            :mu:        params of chebyshev_gaussian
            :emb_file1: if None, emb1 (by RSVD) will not be saved
            :emb_file2: if None, emb2 (by chebyshev) will not be saved
            :smat_file: if None, CSR (model.matrix0) will not be saved
            """
        # BTW: you can also save emb1 and emb2 by:
        from ProNE import save_embedding
        save_embedding(emb_file1, emb1)
        save_embedding(emb_file2, emb1)
        ```
    3. Note:
        you can also load graph from CSR sparse matrix file, CSR file must has suffix `.npz`, like:
        ``` python
        model = ProNE('smat.npz', dimension, node_number=None)
        ```