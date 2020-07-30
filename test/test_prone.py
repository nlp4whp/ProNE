import os
import shutil


def test_train_with_txt(gfile, emb1_f, emb2_f, smat_file):
    try:
        from ProNE import ProNE
        from ProNE import save_embedding
        model = ProNE(gfile)
        emb1, emb2 = model.train(dimension=100, step=10, theta=.5, mu=.2, smat_file=smat_file)
        save_embedding(emb1_f, emb1)
        save_embedding(emb2_f, emb2)
        return True
    except Exception as e:
        return Exception(f"\n>> `test_train_with_txt` ERROR: {e} !!\n")


def test_train_with_csr(gfile, emb1_f, emb2_f, smat_file):
    try:
        from ProNE import ProNE
        from ProNE import save_smat
        model = ProNE(gfile)
        _, _ = model.train(dimension=100, step=10, theta=.5, mu=.2,
                           emb_file1=emb1_f, emb_file2=emb2_f, smat_file=None)
        save_smat(smat_file, model.matrix0)
        return True
    except Exception as e:
        return Exception(f"\n>> `test_train_with_csr` ERROR: {e} !!\n")


if __name__ == "__main__":

    tpath = 'test_code'

    if os.path.isdir(tpath):
        shutil.rmtree(tpath)
    os.mkdir(tpath)

    tlist = {
        'txt': False,
        'csr': False
    }

    tlist['txt'] = test_train_with_txt('../data/PPI.ungraph',
                                       f"{tpath}/emb1_txt",
                                       f"{tpath}/emb2_txt",
                                       f"{tpath}/ppi")

    tlist['csr'] = test_train_with_csr(f'{tpath}/ppi.npz',
                                       f"{tpath}/emb1_csr",
                                       f"{tpath}/emb2_csr",
                                       f"{tpath}/ppi_csr")

    print("\n** final test result **\n")
    for i, (k, e) in enumerate(tlist.items()):
        if e is True:
            print(f"\t# {i}. {k} Success\t")
        else:
            print(f"\t# {i}. {k} ERROR: {e} \t")

    # shutil.rmtree(tpath)
