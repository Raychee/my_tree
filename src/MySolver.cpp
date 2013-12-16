# include <iostream>

# include "MySolver.hpp"

MySolver::MySolver(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
                   SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
                   MyParam&  _my_param):
          SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>(_gd_param, _sgd_param),
          my_param(&_my_param),
          w(NULL),
          b(0),
          p(NULL),
          alloc_p(false),
          margin_x(NULL),
          p_margin(NULL),
          term2(NULL) {
}

MySolver::MySolver(MySolver& some):
          SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>(some),
          my_param(some.my_param),
          w(NULL),
          b(some.b),
          p(some.p),
          alloc_p(false),
          margin_x(some.margin_x),
          p_margin(some.p_margin),
          term2(some.term2) {
    if (some.w) {
        DAT_DIM_T dim = gd_param->dimension();
        w = new COMP_T[dim];
        std::memcpy(w, some.w, dim * sizeof(COMP_T));
    }
}

MySolver::~MySolver() {
    delete[] w;
    if (alloc_p) delete[] p;
}

MySolver& MySolver::ostream_this(std::ostream& out) {
    SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::ostream_this(out);
    out << "\n";
    my_param->ostream_this(out);
    out << "\nTraining parameters:\n\t[w, b] = ";
    ostream_param(out);
    out << "\nProbabilistic distributions ([label|probability]):";
    SUPV_T n_label = stat.num_of_labels();
    for (SUPV_T i = 0; i < n_label; ++i) {
        out << "\n\tLabel " << stat.label(i) << " : " << p[i];
    }
    return *this;
}

MySolver& MySolver::ostream_param(std::ostream& out) {
    DAT_DIM_T dim = gd_param->dimension();
    for (DAT_DIM_T i = 0; i < dim; ++i) {
        out << w[i] << " ";
    }
    out << b;
    return *this;
}

MySolver& MySolver::solve(N_DAT_T* x_pos, N_DAT_T& n_x_pos,
                          N_DAT_T* x_neg, N_DAT_T& n_x_neg) {
    DAT_DIM_T    dim        = gd_param->dimension();
    char         verbosity  = my_param->verbosity();
    unsigned int n_iter     = my_param->num_of_iterations();
    COMP_T       err        = my_param->accuracy();
    bool         show_p     = my_param->show_p_each_iter();
    SUPV_T       s_labelset = stat.num_of_labels();
    N_DAT_T      n_sample   = stat.num_of_samples();
    COMP_T       p_diff;
    if (!w) w = new COMP_T[dim];
    if (!alloc_p) {
        p = new COMP_T[s_labelset];
        alloc_p = true;
    }
    margin_x = new SUPV_T[n];
    p_margin = new COMP_T*[3];
    p_margin[0] = p;
    p_margin[1] = new COMP_T[s_labelset];
    p_margin[2] = new COMP_T[s_labelset];
    term2 = new COMP_T[dim];
    if (verbosity >= 1) {
        std::cout << "MySolver Training: \n\tData: " << n_sample << " samples, "
                  << dim << " feature dimensions.\n\tStopping Criterion: "
                  << n_iter << " iterations or accuracy higher than "
                  << err <<  "." << std::endl;
    }
    if (verbosity == 1) {
        std::cout << "Training ... " << std::flush;
    }
    if (verbosity >= 2) {
        std::cout << "Initializing parameters ... ";
    }
    inititalize(dim, s_labelset, n_sample,
                x_pos, n_x_pos, x_neg, n_x_neg);
    if (verbosity >= 2) {
        std::cout << "Done.";
        if (show_p) {
            std::cout << "\n\t";
            for (SUPV_T i = 0; i < s_labelset; ++i) {
                std::cout << "[" << stat.label(i) << "|" << p[i] << "]";
            }
        }
        std::cout << std::endl;
    }
    unsigned int i;
    for (i = 0; i < n_iter; ++i) {
        if (verbosity >= 2) {
            std::cout << "My Solver Training: Iteration " << i + 1 << " ... ";
            if (verbosity >= 3) {
                std::cout << "\n\tTraining SVM ... ";
            }
            std::cout << std::endl;
        }
        t = 0;
        train();
        if (verbosity >= 3) {
            std::cout << "Done.\n\tUpdating p ... " << std::flush;
        }
        p_diff = update_p(dim, x_pos, n_x_pos, x_neg, n_x_neg);
        if (verbosity >= 2) {
            std::cout << "Done.";
            if (show_p) {
                std::cout << "\n\t";
                for (SUPV_T i = 0; i < s_labelset; ++i) {
                    std::cout << "[" << stat.label(i) << "|" << p[i] << "]";
                }
            }
            std::cout << std::endl;
        }
        if (p_diff < err) break;
    }
    if (verbosity == 1) {
        std::cout << "Done." << std::endl;
    }
    delete[] margin_x;
    delete[] p_margin[1];
    delete[] p_margin[2];
    delete[] p_margin;
    delete[] term2;
    if (verbosity >= 1) {
        if (i < n_iter)
            std::cout << "Training stopped at iteration " << i + 1 << " with convergence.";
        else
            std::cout << "Max number of iterations has been reached.";
        std::cout << "\nMySolver Training: finished." << std::endl;
    }
    return *this;
}

COMP_T MySolver::update_p(DAT_DIM_T d,
                          N_DAT_T* x_pos, N_DAT_T& n_x_pos,
                          N_DAT_T* x_neg, N_DAT_T& n_x_neg) {
    SUPV_T s_labelset = stat.num_of_labels();
    COMP_T diff_p = 0; 
    n_x_pos = n_x_neg = 0;
    for (SUPV_T i = 0; i < s_labelset; ++i) {
        N_DAT_T* x_label   = stat[i];
        N_DAT_T  n_x_label = stat.num_of_samples_with_label(i);
        N_DAT_T  n_x_label_pos = 0;
        for (N_DAT_T j = 0; j < n_x_label; ++j) {
            N_DAT_T x_i = x_label[j];
            COMP_T* dat_i = data + d * x_i;
            COMP_T  score = compute_score(dat_i, d);
            if (score > 0) {
                ++n_x_label_pos;
                x_pos[n_x_pos++] = x_i;
            }
            else { x_neg[n_x_neg++] = x_i;}
            if (score >= 1) { margin_x[x_i] = 2; }
            else if (score <= -1) { margin_x[x_i] = 0; }
            else { margin_x[x_i] = 1; }
        }
        COMP_T p_new = (COMP_T)n_x_label_pos / n_x_label;
        COMP_T p_diff = p[i] - p_new;
        if (p_diff < 0) p_diff = -p_diff;
        diff_p += p_diff;
        p[i] = p_new;
        p_margin[1][i] = 2 * p[i] - 1;
        p_margin[2][i] = p[i] - 1;
    }
    return diff_p;
}

inline MySolver& MySolver::train_iteration(COMP_T*   data,
                                           DAT_DIM_T d,
                                           N_DAT_T   n,
                                           SUPV_T*   y) {
    COMP_T term1 = 1 - eta * my_param->regul_coef();
    for (DAT_DIM_T i = 0; i < d; ++i) {
        term2[i] = 0;
        w[i] *= term1;
    }
    COMP_T  term2_b = 0;
    COMP_T  term3   = eta / n;
    COMP_T* dat_i   = data;
    for (N_DAT_T i = 0; i < n; ++i, dat_i += d) {
        COMP_T p_i  = p_margin[margin_x[i]][stat.index_of_label(y[i])];
        for (DAT_DIM_T i = 0; i < d; ++i) {
            term2[i] += p_i * dat_i[i];
        }
        term2_b += p_i;
    }
    for (DAT_DIM_T i = 0; i < d; ++i) {
        w[i] += term3 * term2[i];
    }
    b += term3 * term2_b;
    return *this;
}
MySolver& MySolver::train_iteration(COMP_T*   data,
                                    DAT_DIM_T d,
                                    N_DAT_T*  x,
                                    N_DAT_T   n,
                                    SUPV_T*   y) {
    COMP_T term1 = 1 - eta * my_param->regul_coef();
    for (DAT_DIM_T i = 0; i < d; ++i) {
        term2[i] = 0;
        w[i] *= term1;
    }
    COMP_T term2_b = 0;
    COMP_T term3 = eta / n;
    for (N_DAT_T i = 0; i < n; ++i) {
        N_DAT_T x_i   = x[i];
        COMP_T* dat_i = data + d * x_i;
        COMP_T  p_i   = p_margin[margin_x[x_i]][stat.index_of_label(y[x_i])];
        for (DAT_DIM_T i = 0; i < d; ++i) {
            term2[i] += p_i * dat_i[i];
        }
        term2_b += p_i;
    }
    for (DAT_DIM_T i = 0; i < d; ++i) {
        w[i] += term3 * term2[i];
    }
    b += term3 * term2_b;
    return *this;
}

// Using Hinge Loss
COMP_T MySolver::compute_obj(COMP_T*   data,
                             DAT_DIM_T d,
                             N_DAT_T   n,
                             SUPV_T*   y) {
    COMP_T loss  = 0;
    COMP_T* dat_i = data;
    for (N_DAT_T i = 0; i < n; ++i, dat_i += d) {
        COMP_T  p_y      = p[stat.index_of_label(y[i])];
        COMP_T  score    = compute_score(dat_i, d);
        COMP_T  pos_loss = score < 1 ? 1 - score : 0;
        COMP_T  neg_loss = score > -1 ? 1 + score : 0;
        loss += p_y * pos_loss + (1 - p_y) * neg_loss;
    }
    return loss / n + my_param->regul_coef() * compute_norm(d);
}
// Using Hinge Loss
COMP_T MySolver::compute_obj(COMP_T*   data,
                             DAT_DIM_T d,
                             N_DAT_T*  x,
                             N_DAT_T   n,
                             SUPV_T*   y) {
    COMP_T loss = 0;
    for (N_DAT_T i = 0; i < n; ++i) {
        N_DAT_T x_i      = x[i];
        COMP_T  p_y      = p[stat.index_of_label(y[x_i])];
        COMP_T  score    = compute_score(data + d * x_i, d);
        COMP_T  pos_loss = score < 1 ? 1 - score : 0;
        COMP_T  neg_loss = score > -1 ? 1 + score : 0;
        loss += p_y * pos_loss + (1 - p_y) * neg_loss;
    }
    return loss / n + my_param->regul_coef() * compute_norm(d);
}

MySolver& MySolver::train_one(COMP_T*   data,
                              DAT_DIM_T d,
                              N_DAT_T   i,
                              N_DAT_T   n,
                              SUPV_T*   y) {
    COMP_T term1 = 1 - eta * my_param->regul_coef();
    for (DAT_DIM_T i = 0; i < d; ++i) {
        w[i] *= term1;
    }
    COMP_T term2 = eta / n;
    COMP_T  p_i   = p_margin[margin_x[i]][stat.index_of_label(y[i])];
    COMP_T* dat_i = data + d * i;
    for (DAT_DIM_T i = 0; i < d; ++i) {
        w[i] += term2 * p_i * dat_i[i];
    }
    b += term2 * p_i;
    return *this;
}

MySolver& MySolver::inititalize(DAT_DIM_T d,
                                SUPV_T    n_label,
                                N_DAT_T   n_sample,
                                N_DAT_T* x_pos, N_DAT_T& n_x_pos,
                                N_DAT_T* x_neg, N_DAT_T& n_x_neg) {
    // N_DAT_T n_subsample = gd_param->learning_rate_sample_rate() * n_sample;
    // N_DAT_T x = new N_DAT_T[n_subsample];
    // stat.rand_index(x, n_subsample);
    // for (DAT_DIM_T i = 0; i < d; ++i) term2[i] = 0;
    // for (N_DAT_T i = 0; i < n_subsample; ++i) {
    //     COMP_T* dat_i = data + d * x[i];
    //     for (DAT_DIM_T i = 0; i < d; ++i) {
    //         term2[i] += dat_i[i];
    //     }
    // }
    // for (DAT_DIM_T i = 0; i < d; ++i) {
    //     term2[i] /= n_subsample;

    // }
    for (DAT_DIM_T i = 0; i < d; ++i) {
        w[i] = 0;
    }
    b = 0;
    update_p(d, x_pos, n_x_pos, x_neg, n_x_neg);
    SUPV_T half_n_label = n_label / 2;
    SUPV_T i;
    for (i = 0; i < half_n_label; ++i) p[i] = 1;
    for (; i < n_label; ++i) p[i] = 0;
    for (SUPV_T i = 0; i < n_label; ++i) {
        SUPV_T temp_i = std::rand() % (n_label - i) + i;
        COMP_T temp   = p[temp_i];
        p[temp_i] = p[i];
        p[i] = temp;
    }
    // delete[] x;
    return *this;
}

COMP_T MySolver::compute_score(COMP_T* dat_i, DAT_DIM_T d) const {
    COMP_T score = 0;
    for (DAT_DIM_T i = 0; i < d; ++i) {
        score += w[i] * dat_i[i];
    }
    return score + b;
}

COMP_T MySolver::compute_norm(DAT_DIM_T d) const {
    COMP_T norm = 0;
    for (DAT_DIM_T i = 0; i < d; ++i) {
        COMP_T w_i = w[i];
        norm += w_i * w_i;
    }
    return norm;
}
