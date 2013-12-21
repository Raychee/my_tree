# include <iostream>
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
               float        _eta0_try_sample_rate,
               COMP_T       _eta0_try_1st,
               COMP_T       _eta0_try_factor,
               bool         _show_obj_each_iter):
        gd_param(_dimension,
                 _verbosity - 1,
                 _eta0,
                 200,
                 1e-8,
                 _eta0_try_sample_rate,
                 _eta0_try_1st,
                 _eta0_try_factor,
                 _show_obj_each_iter),
        sgd_param(_s_batch,
                  _n_epoch),
        my_param(_verbosity,
                 _lambda,
                 _n_iter,
                 _n_iter_fine,
                 _err,
                 _show_p_each_iter),
        min_entropy_(_min_entropy),
        max_depth_(_max_depth) {
}

MyTree& MyTree::train(COMP_T* data, N_DAT_T n, SUPV_T* y,
                      N_DAT_T* x, N_DAT_T s_x) {  
    MySolver* solver;
    solver = new MySolver(gd_param, sgd_param, my_param);
    solver->training_data(data, n, y, x, s_x);
    if (!s_x) s_x = n;
    N_DAT_T* x_pos = new N_DAT_T[s_x];
    N_DAT_T* x_neg = new N_DAT_T[s_x];
    N_DAT_T  n_x_pos;
    N_DAT_T  n_x_neg;
    root = new TreeNode(solver);
    iterator it_end = end();
    for (iterator i = begin(); i != it_end; ++i) {
        TreeNode* node;
        solver = i->pcontent();
        if (max_depth_ > 0 && i->depth() >= max_depth_) continue;
        if (solver->entropy() < min_entropy_) continue;
        solver->solve(x_pos, n_x_pos, x_neg, n_x_neg);
        solver = new MySolver(gd_param, sgd_param, my_param);
        solver->training_data(data, n, y, x_pos, n_x_pos);
        node = new TreeNode(solver);
        i->attach_child(node);
        solver = new MySolver(gd_param, sgd_param, my_param);
        solver->training_data(data, n, y, x_neg, n_x_neg);
        node = new TreeNode(solver);
        i->attach_child(node);
    }    
    return *this;
}

MyTree& MyTree::test(COMP_T* data, N_DAT_T n, SUPV_T* y) {
    DAT_DIM_T dim   = gd_param.dimension();
    COMP_T*   dat_i = data;
    for (N_DAT_T i = 0; i < n; ++i, dat_i += dim) {
        y[i] = test_one(dat_i);
    }
    return *this;
}

SUPV_T MyTree::test_one(COMP_T* data) {
    LabelStat<SUPV_T, N_DAT_T>& distrib = test_distrib(data);
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


LabelStat<SUPV_T, N_DAT_T>& MyTree::test_distrib(COMP_T* data) {
    DAT_DIM_T dim    = gd_param.dimension();
    TreeNode& node   = *root;
    MySolver& solver = node.content();
    while (node.has_child()) {
        node   = node.child(solver.test_one(data, dim));
        solver = node.content();
    }
    return solver.distribution();
}

