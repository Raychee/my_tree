# include <iostream>
# include "MyTree.hpp"


MySolver::MySolver(Param&   _param,
                   MyParam& _my_param,
                   N_DAT_T* _x,
                   N_DAT_T* _s_x,
                   N_DAT_T* _d):
          SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>(_param),
          my_param(&_my_param),
          b(0),
          x(_x),
          s_x(_s_x) {
    w = new COMP_T[param->dimension()];
    SUPV_T _s_labelset = my_param->num_of_labels();
    d.insert(_d, _s_labelset);
}

MySolver::MySolver(MySolver& some):
          SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>(some),
          my_param(some.my_param),
          b(some.b),
          x(some.x),
          s_x(some.s_x),
          d(some.d),
          p(some.p) {
    if (some.w) {
        DAT_DIM_T dim = param->dimension();
        w = new COMP_T[dim];
        std::memcpy(w, some.w, dim * sizeof(COMP_T));
    }
}

MySolver::~MySolver() {
    if (w) delete[] w;
    if (x) delete[] x;
}

COMP_T MySolver::compute_obj(COMP_T* dat, N_DAT_T n, SUPV_T* y) {
    DAT_DIM_T dim    = param->dimension();
    N_DAT_T   x_size = x->size();
    COMP_T    loss   = 0;
    for (N_DAT_T i = 0; i < x_size; ++i) {
        N_DAT_T x_i      = (*x)[i];
        SUPV_T  y_i      = y[x_i];
        COMP_T  p_y_i    = p[y_i];
        COMP_T  score    = compute_score(dat + dim * x_i);
        COMP_T  pos_loss = score < 1 ? 1 - score : 0;
        COMP_T  neg_loss = score > -1 ? 1 + score : 0;
        loss += p_y_i * pos_loss + (1 - p_y_i) * neg_loss;
    }
    return loss + my_param->regul_coef() * compute_norm();
}

MySolver& MySolver::ostream_this(std::ostream& out) {
    return *this;
}

MySolver& MySolver::ostream_param(std::ostream& out) {
    return *this;
}

MySolver& MySolver::solve(COMP_T* dat, N_DAT_T n, SUPV_T* y,
                          N_DAT_T* x0 = NULL, N_DAT_T* x1 = NULL,
                          N_DAT_T* d0 = NULL, N_DAT_T* d1 = NULL) {
    bool temp_buf_x0 = !x0, temp_buf_x1 = !x1,
         temp_buf_d0 = !d0, temp_buf_d0 = !d1;
    SUPV_T s_labelset = my_param->num_of_labels();
    if (temp_buf_x0) x0 = new N_DAT_T[n];
    if (temp_buf_x1) x1 = new N_DAT_T[n];
    if (temp_buf_d0) x0 = new N_DAT_T[s_labelset];
    if (temp_buf_d1) x1 = new N_DAT_T[s_labelset];
    
}

MySolver& MySolver::update_p(N_DAT_T* d0) {

}

MySolver& MySolver::train_one(COMP_T* dat, N_DAT_T i,
                              N_DAT_T n, SUPV_T* y) {
    return *this;
}

SUPV_T MySolver::test_one(COMP_T* dat_i) const {

}

COMP_T MySolver::compute_learning_rate() {
    COMP_T eta0   = param->init_learning_rate();
    COMP_T lambda = my_param->regul_coef();
    return eta0 / (1 + lambda * eta0 * t);
}

COMP_T MySolver::compute_score(COMP_T* dat_i) const {
    DAT_DIM_T dim   = param->dimension();
    COMP_T    score = 0;
    for (DAT_DIM_T i = 0; i < dim; ++i) {
        score += w[i] * dat_i[i];
    }
    return score + b;
}

// Using Hinge Loss
COMP_T MySolver::compute_loss(COMP_T* dat_i, unsigned int y) const {
    COMP_T margin = compute_score(dat_i);
    if (y) margin = -margin;
    if (margin < 1) return 1 - margin;
    else return 0;
}

COMP_T MySolver::compute_norm() const {
    DAT_DIM_T dim  = param->dimension();
    COMP_T    norm = 0;
    for (DAT_DIM_T i = 0; i < dim; ++i) {
        norm += w[i] * w[i];
    }
    return norm;
}


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

