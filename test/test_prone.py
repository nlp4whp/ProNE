import os
import shutil


def test_run(gfile, emb1, emb2, smat_file):
    try:
        from ProNE import run
        run(gfile, emb1, emb2, smat_file, node_num=None, dim=100, step=10, theta=0.5, mu=0.2)
        return True
    except Exception as e:
        return Exception(f"\n>> test run ERROR: {e} !!\n")


def test_fn(gfile, emb1, emb2, smat_file):
    try:
        from ProNE import ProNE
        from ProNE import save_smat
        from ProNE import save_embedding

        model = ProNE(gfile, dimension=100)
        save_smat(smat_file, model.matrix0)
        features_matrix = model.pre_factorization(model.matrix0, model.matrix0)
        save_embedding(emb1, features_matrix)
        embeddings_matrix = model.chebyshev_gaussian(
                                        A=model.matrix0,
                                        a=features_matrix,
                                        order=10,
                                        mu=.5,
                                        s=.2)
        save_embedding(emb2, embeddings_matrix)
        return True
    except Exception as e:
        return Exception(f"\n>> test functions ERROR: {e} !!\n")


def test_csr(gfile, emb1, emb2, csr_file):
    try:
        from ProNE import run
        from ProNE import ProNE
        from ProNE import save_smat
        model = ProNE(gfile, dimension=100)
        save_smat(csr_file, model.matrix0)
        model = ProNE(f"{csr_file}.npz", dimension=100)
        run(f"{csr_file}.npz", emb1, emb2, node_num=None, dim=100, step=10, theta=0.5, mu=0.2)
        return True
    except Exception as e:
        return Exception(f"\n>> test csr ERROR: {e} !!\n")


if __name__ == "__main__":

    tpath = 'test_code'

    if os.path.isdir(tpath):
        shutil.rmtree(tpath)
    os.mkdir(tpath)

    tlist = {
        'run': False,
        'function': False,
        'csr': False
    }

    tlist['run'] = test_run('../data/PPI.ungraph',
                            f"{tpath}/emb1",
                            f"{tpath}/emb2",
                            f"{tpath}/ppi")

    tlist['function'] = test_fn('../data/cornell.ungraph',
                                f"{tpath}/emb1",
                                f"{tpath}/emb2",
                                f"{tpath}/cornell")

    tlist['csr'] = test_csr('../data/test.ungraph',
                            f"{tpath}/emb1",
                            f"{tpath}/emb2",
                            f"{tpath}/test")

    print("\n** final test result **\n")
    for i, (k, e) in enumerate(tlist.items()):
        if e is True:
            print(f"\t# {i}. test {k} Success\t")
        else:
            print(f"\t# {i}. test {k} ERROR: {e} \t")

    shutil.rmtree(tpath)
