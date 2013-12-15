# include <iostream>
# include "MyTree.hpp"

MyTree::MyTree(DAT_DIM_T    _dimension,
               SUPV_T       _s_labelset,
               COMP_T       _lambda,
               char         _verbosity,
               COMP_T       _eta0,
               unsigned int _n_epoch,
               COMP_T       _eta0_1st_try,
               COMP_T       _eta0_try_factor,
               bool         _comp_obj_each_epoch):
        sgd_param(_dimension,
                  _verbosity,
                  _eta0,
                  _n_epoch,
                  _eta0_1st_try,
                  _eta0_try_factor,
                  _comp_obj_each_epoch),
        my_param(_s_labelset, _lambda) {
}

MyTree& MyTree::train(COMP_T* dat, N_DAT_T n, SUPV_T* y) {
    SUPV_T s_labelset = my_param.num_of_labels();
    std::vector<N_DAT_T>* x = new std::vector<N_DAT_T>;
    x->reserve(n);
    N_DAT_T* d = new N_DAT_T[s_labelset];
    std::memset(d, 0, s_labelset * sizeof(N_DAT_T));
    for (N_DAT_T i = 0; i < n; ++i) {
        x->push_back(i);
        ++d[y[i]];
    }
    MySolver* solver = new MySolver(sgd_param, my_param, x, d);
    root = new TreeNode(solver);
    
    delete[] d;
    return *this;
}

