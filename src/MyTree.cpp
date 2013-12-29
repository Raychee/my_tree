# include <iostream>
# include <iomanip>
# include "MyTree.hpp"


MyTree::MyTree(char         _verbosity,
               DAT_DIM_T    _dimension,
               COMP_T       _min_entropy,
               SUPV_T       _max_depth,
               COMP_T       _lambda,
               unsigned int _n_iter,
               unsigned int _n_iter_fine,
               COMP_T       _err,
               bool         _show_p_each_iter,
               COMP_T       _eta0,
               N_DAT_T      _s_batch,
               unsigned int _n_epoch,
               N_DAT_T      _min_n_subsample,
               float        _eta0_try_sample_rate,
               COMP_T       _eta0_try_1st,
               COMP_T       _eta0_try_factor,
               bool         _show_obj_each_iter):
        alloc(true) {
    gd_param      = new GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::
                        GDParam(_dimension,
                                _verbosity - 1,
                                _eta0,
                                200,
                                1e-8,
                                _min_n_subsample,
                                _eta0_try_sample_rate,
                                _eta0_try_1st,
                                _eta0_try_factor,
                                _show_obj_each_iter);
    sgd_param     = new SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::
                        SGDParam(_s_batch, _n_epoch);
    my_param      = new MySolver::MyParam(_verbosity,
                                          _lambda,
                                          _n_iter,
                                          _n_iter_fine,
                                          _err,
                                          _show_p_each_iter);
    my_tree_param = new MyTreeParam(_min_entropy, _max_depth);
}

MyTree::MyTree(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
               SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
               MySolver::MyParam&                                 _my_param,
               MyTreeParam&                                       _my_tree_param):
        gd_param(&_gd_param),
        sgd_param(&_sgd_param),
        my_param(&_my_param),
        my_tree_param(&_my_tree_param),
        alloc(false) {
}

MyTree::~MyTree() {
    if (alloc) {
        delete gd_param;
        delete sgd_param;
        delete my_param;
        delete my_tree_param;
    }
}

MyTree& MyTree::train(COMP_T* data, N_DAT_T n, SUPV_T* y,
                      N_DAT_T* x, N_DAT_T s_x) {
    COMP_T        min_entropy = my_tree_param->min_entropy();
    SUPV_T        max_depth   = my_tree_param->max_depth();
    std::ostream* out         = my_tree_param->ostream_of_training_result();
    gd_param->init_learning_rate(gd_param->learning_rate_1st_try());
    MySolver* solver;
    solver = new MySolver(*gd_param, *sgd_param, *my_param);
    solver->training_data(data, n, y, x, s_x);
    if (!x || !s_x) s_x = n;
    N_DAT_T* x_pos = new N_DAT_T[s_x];
    N_DAT_T* x_neg = new N_DAT_T[s_x];
    N_DAT_T  n_x_pos;
    N_DAT_T  n_x_neg;
    root = new TreeNode(solver);
    iterator it_end = end();
    for (iterator i = begin(); i != it_end; ++i) {
        TreeNode* node;
        solver = i->pcontent();
        if (max_depth > 0 && i->depth() >= (unsigned)max_depth) continue;
        if (solver->entropy() < min_entropy) continue;
        gd_param->learning_rate_1st_try(gd_param->init_learning_rate());
        gd_param->init_learning_rate(0);
        solver->solve(x_pos, n_x_pos, x_neg, n_x_neg);
        if (out) {
            solver->ostream_param(*out);
            *out << std::endl;
        }
        solver = new MySolver(*gd_param, *sgd_param, *my_param);
        solver->training_data(data, n, y, x_pos, n_x_pos);
        node = new TreeNode(solver);
        i->attach_child(node);
        solver = new MySolver(*gd_param, *sgd_param, *my_param);
        solver->training_data(data, n, y, x_neg, n_x_neg);
        node = new TreeNode(solver);
        i->attach_child(node);
    }
    delete[] x_pos;
    delete[] x_neg;
    return *this;
}

MyTree& MyTree::test(COMP_T* data, N_DAT_T n, SUPV_T* y) {
    DAT_DIM_T dim   = gd_param->dimension();
    COMP_T*   dat_i = data;
    for (N_DAT_T i = 0; i < n; ++i, dat_i += dim) {
        y[i] = test_one(dat_i, dim);
    }
    return *this;
}

SUPV_T MyTree::test_one(COMP_T* data, DAT_DIM_T dim) {
    LabelStat<SUPV_T, N_DAT_T>& distrib = test_distrib(data, dim);
    SUPV_T                      n_label = distrib.num_of_labels();
    N_DAT_T            max_n_x_of_label = 0;
    SUPV_T               max_n_of_label = 0;
    for (SUPV_T i = 0; i < n_label; ++i) {
        N_DAT_T n_x_of_label = distrib.num_of_samples_with_label(i);
        if (n_x_of_label > max_n_x_of_label) {
            max_n_x_of_label = n_x_of_label;
            max_n_of_label = i;
        }
    }
    return distrib.label(max_n_of_label);
}

LabelStat<SUPV_T, N_DAT_T>& MyTree::test_distrib(COMP_T* data, DAT_DIM_T dim) {
    TreeNode* node   = root;
    MySolver* solver = node->pcontent();
    while (node->has_child()) {
        node   = node->pchild(solver->test_one(data, dim));
        solver = node->pcontent();
    }
    return solver->distribution();
}
