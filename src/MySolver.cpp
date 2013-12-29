# include <iostream>
# include <limits>
# include <cmath>

# include "MySolver.hpp"

MySolver::MySolver(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
                   SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
                   char         _v,
                   COMP_T       _lambda,
                   unsigned int _n_trial,
                   unsigned int _n_train,
                   unsigned int _n_iter,
                   unsigned int _n_iter_fine,
                   COMP_T       _err,
                   bool         _show_p_each_iter):
          SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>(_gd_param, _sgd_param),
          w(NULL),
          b(0),
          p(NULL),
          alloc_my_param(true),
          x_pos(NULL),
          n_x_pos(0),
          x_neg(NULL),
          n_x_neg(0),
          p_margin_pos(NULL),
          p_margin_mid(NULL),
          p_margin_neg(p),
          term2(NULL) {
    my_param = new MyParam(_v,
                           _lambda,
                           _n_trial,
                           _n_train,
                           _n_iter,
                           _n_iter_fine,
                           _err,
                           _show_p_each_iter);
}

MySolver::MySolver(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
                   SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
                   MyParam&  _my_param):
          SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>(_gd_param, _sgd_param),
          my_param(&_my_param),
          w(NULL),
          b(0),
          p(NULL),
          alloc_my_param(false),
          x_pos(NULL),
          n_x_pos(0),
          x_neg(NULL),
          n_x_neg(0),
          p_margin_pos(NULL),
          p_margin_mid(NULL),
          p_margin_neg(p),
          term2(NULL) {
}

MySolver::MySolver(MySolver& some):
          SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>(some),
          my_param(some.my_param),
          w(NULL),
          b(some.b),
          p(NULL),
          alloc_my_param(false),
          x_pos(some.x_pos),
          n_x_pos(some.n_x_pos),
          x_neg(some.x_neg),
          n_x_neg(some.n_x_neg),
          p_margin_pos(NULL),
          p_margin_mid(NULL),
          p_margin_neg(p),
          term2(NULL) {
    DAT_DIM_T dim     = gd_param->dimension();
    SUPV_T    n_label = stat.num_of_labels();
    if (some.w) {
        w = new COMP_T[dim];
        std::memcpy(w, some.w, dim * sizeof(COMP_T));
    }
    if (some.p) {
        p = new COMP_T[n_label];
        std::memcpy(p, some.p, n_label * sizeof(COMP_T));
    }
    if (some.p_margin_pos) {
        p_margin_pos = new COMP_T[n_label];
        std::memcpy(p_margin_pos, some.p_margin_pos, n_label * sizeof(COMP_T));
    }
    if (some.p_margin_mid) {
        p_margin_mid = new COMP_T[n_label];
        std::memcpy(p_margin_mid, some.p_margin_mid, n_label * sizeof(COMP_T));
    }
    if (some.term2) {
        term2 = new COMP_T[n_label];
        std::memcpy(term2, some.term2, n_label * sizeof(COMP_T));
    }
}

MySolver::~MySolver() {
    if (alloc_my_param) delete my_param;
    delete[] w;
    delete[] p;
    delete[] p_margin_pos;
    delete[] p_margin_mid;
    delete[] term2;
}

MySolver& MySolver::ostream_this(std::ostream& out) {
    stat.ostream_this(out);
    out << "\n";
    // SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::ostream_this(out);
    // out << "\n";
    // my_param->ostream_this(out);
    if (w && p) {
        out << "\nTraining parameters:\n    [w, b] = ";
        ostream_param(out);
        out << "\nProbabilistic distributions ([label|probability]):\n    ";
        SUPV_T n_label = stat.num_of_labels();
        for (SUPV_T i = 0; i < n_label; ++i) {
            out << "[ " << stat.label(i) << " | " << p[i] << " ]";
        }
    }
    return *this;
}

MySolver& MySolver::ostream_param(std::ostream& out) {
    if (!w) return *this;
    DAT_DIM_T dim = gd_param->dimension();
    for (DAT_DIM_T i = 0; i < dim; ++i) {
        out << w[i] << " ";
    }
    out << b;
    return *this;
}

MySolver& MySolver::solve(N_DAT_T* _x_pos, N_DAT_T& _n_x_pos,
                          N_DAT_T* _x_neg, N_DAT_T& _n_x_neg) {
    x_pos = _x_pos;
    x_neg = _x_neg;
    solve();
    _n_x_pos = n_x_pos;
    _n_x_neg = n_x_neg;
    return *this;
}

MySolver& MySolver::solve() {
    DAT_DIM_T     dim         = gd_param->dimension();
    char          gd_v        = gd_param->verbosity();
    N_DAT_T       n_subsample = gd_param->min_num_of_subsamples();
    COMP_T        eta0        = gd_param->init_learning_rate();
    char          my_v        = my_param->verbosity();
    unsigned int  n_trial     = my_param->num_of_trials();
    unsigned int  n_train     = my_param->num_of_trainings();
    unsigned int  n_iter      = my_param->num_of_iterations();
    unsigned int  n_iter_fine = my_param->num_of_fine_tuning();
    COMP_T        err         = my_param->accuracy();
    bool          show_p      = my_param->show_p_each_iter();
    SUPV_T        n_label     = stat.num_of_labels();
    N_DAT_T       n_sample    = stat.num_of_samples();
    N_DAT_T*      rand_x;
    N_DAT_T*      x_subsample;
    COMP_T*       center;
    COMP_T        info_gain_opt = 0;
    MySolver*     best_solver = NULL;
    if (!w) w = new COMP_T[dim];
    if (!p) p = new COMP_T[n_label];
    p_margin_pos = new COMP_T[n_label];
    p_margin_mid = new COMP_T[n_label];
    term2        = new COMP_T[dim];
    rand_x       = new N_DAT_T[n_sample];
    stat.index_of_samples(rand_x);

    if (my_v >= 1) {
        std::cout << "MySolver Training: \n    Data: " << n_sample << " samples, "
                  << dim << " feature dimensions.\n    Stopping Criterion: "
                  << n_iter << " full iterations and " << n_iter_fine 
                  << " fine-tuning iterations";
        if (err > 0) std::cout << " or accuracy higher than " << err;
        std::cout <<  ".\nMySolver Training: begin." << std::endl;
    }
    
    if (!n_subsample) n_subsample = n_sample;
    x_subsample = new N_DAT_T[n_subsample];
    center      = new COMP_T[dim];
    stat.rand_index(x_subsample, n_subsample);
    for (DAT_DIM_T i = 0; i < dim; ++i) center[i] = 0;
    for (N_DAT_T i = 0; i < n_subsample; ++i) {
        COMP_T* dat_i = data + dim * x_subsample[i];
        for (DAT_DIM_T i = 0; i < dim; ++i) center[i] += dat_i[i];
    }
    for (DAT_DIM_T i = 0; i < dim; ++i) center[i] /= n_subsample;

    t = 0;
    for (unsigned int i_trial = 0; i_trial < n_trial; ++i_trial) {
        if (my_v >= 1) {
            std::cout << "Trying parameters " << i_trial + 1 << " ... \n"
                      << "Initializing ... " << std::flush;
        }
        COMP_T* sample = data + dim * x_subsample[i_trial];
        b = 0;
        for (DAT_DIM_T i = 0; i < dim; ++i) {
            w[i] = sample[i] - center[i];
            b -= w[i] * center[i];
        }
        update_p(dim, n_label);
        if (my_v >= 1) {
            std::cout << "Done.";
            if (show_p) {
                std::cout << "\n    p = ";
                for (SUPV_T i = 0; i < n_label; ++i) {
                    std::cout << "[" << stat.label(i) << "|" << p[i] << "]";
                }
            }
            std::cout << std::endl;
        }
        for (unsigned int i_train = 0; i_train < n_train; ++i_train) {
            if (my_v >= 1) {
                std::cout << "Training " << i_train + 1 << " ... ";
                if (my_v > 1 || gd_v >= 1) std::cout << std::endl;
                else std::cout.flush();
            }
            unsigned int i, j;
            gd_param->init_learning_rate(eta0);
            MySolver* try_solver = new MySolver(*this);
            try_solver->full_train(i, n_iter, dim, rand_x, n_sample, n_label);
            if (my_v == 1 && gd_v < 1) std::cout << "Done.";
            // if (my_v >= 1) {
                
            //     if (my_v > 1 || gd_v >= 1) std::cout << std::endl;
            //     else std::cout.flush();
            // }
            if (i >= n_iter) {
                if (my_v >= 1) {
                    std::cout << "Fine-tuning ... ";
                    if (my_v == 1 && gd_v < 1) std::cout.flush();
                    else std::cout << std::endl;
                }
                try_solver->fine_train(j, n_iter_fine, dim, rand_x, n_sample, n_label);
                if (my_v == 1 && gd_v < 1) std::cout << "Done.\n";
            }
            if (my_v >= 1) {
                std::cout << "Training " << i_train + 1 << ": finished. \n";
                if (i < n_iter)
                    std::cout << "    Training stopped at iteration " << i + 1 
                              << " with convergence.";
                else if (j < n_iter_fine) {
                    std::cout << "    Training stopped during fine-tuning iteration "
                              << j + 1 << " with convergence.";
                }
                else std::cout << "    Max number of iterations has been reached.";
                std::cout << std::endl;
            }
            COMP_T info_gain = try_solver->info_gain(n_sample, n_label);
            if (info_gain > info_gain_opt) {
                delete best_solver;
                best_solver = try_solver;
                info_gain_opt = info_gain;
                if (my_v >= 1) {
                    std::cout << "    Current result is the best so far."
                              << std::endl;
                }
            }
            else {
                delete try_solver;
                if (my_v >= 1) {
                    std::cout << "    Current result is not the best."
                              << std::endl;
                }
            }
        }
    }
    if (best_solver) {
        std::memcpy(w, best_solver->w, dim * sizeof(COMP_T));
        b = best_solver->b;
        std::memcpy(p, best_solver->p, n_label * sizeof(COMP_T));
        if (x_pos && x_neg) update_p(dim, n_label);
    }
    
    delete[] p_margin_mid; p_margin_mid = NULL;
    delete[] p_margin_pos; p_margin_pos = NULL;
    delete[] term2;        term2        = NULL;
    delete[] rand_x;
    delete[] x_subsample;
    delete[] center;
    delete   best_solver;
    if (my_v >= 1) {
        std::cout << "MySolver Training: finished. \n";
    }
    return *this;
}

COMP_T MySolver::update_p(DAT_DIM_T d, SUPV_T n_label) {
    COMP_T diff_p = 0; 
    n_x_pos = n_x_neg = 0;
    for (SUPV_T i = 0; i < n_label; ++i) {
        N_DAT_T* x_label   = stat[i];
        N_DAT_T  n_x_label = stat.num_of_samples_with_label(i);
        N_DAT_T  n_x_label_pos = 0;
        for (N_DAT_T j = 0; j < n_x_label; ++j) {
            N_DAT_T x_i = x_label[j];
            COMP_T* dat_i = data + d * x_i;
            COMP_T  score = compute_score(dat_i, d);
            if (score > 0) {
                ++n_x_label_pos;
                if (x_pos) x_pos[n_x_pos++] = x_i;
            }
            else if (x_neg) x_neg[n_x_neg++] = x_i;
        }
        COMP_T p_new = (COMP_T)n_x_label_pos / n_x_label;
        COMP_T p_diff = p[i] - p_new;
        diff_p += p_diff >= 0 ? p_diff : -p_diff;
        p[i] = p_new;
        p_margin_mid[i] = 2 * p[i] - 1;
        p_margin_pos[i] = p[i] - 1;
    }
    return diff_p;
}

// MySolver& MySolver::train_batch(COMP_T*   data,
//                                 DAT_DIM_T d,
//                                 N_DAT_T   n,
//                                 SUPV_T*   y,
//                                 COMP_T    eta) {
//     COMP_T term1 = 1 - eta * my_param->regul_coef();
//     for (DAT_DIM_T i = 0; i < d; ++i) {
//         term2[i] = 0;
//     }
//     COMP_T  term2_b = 0;
//     COMP_T  term3   = eta / n;
//     COMP_T* dat_i   = data;
//     for (N_DAT_T i = 0; i < n; ++i, dat_i += d) {
//         COMP_T  score = compute_score(dat_i, d);
//         COMP_T  p_i;
//         if (score < -1) p_i = p_margin_neg[stat.index_of_label(y[i])];
//         else if (score > 1) p_i = p_margin_pos[stat.index_of_label(y[i])];
//         else p_i = p_margin_mid[stat.index_of_label(y[i])];
//         for (DAT_DIM_T i = 0; i < d; ++i) {
//             term2[i] += p_i * dat_i[i];
//         }
//         term2_b += p_i;
//     }
//     for (DAT_DIM_T i = 0; i < d; ++i) {
//         w[i] = term1 * w[i] + term3 * term2[i];
//     }
//     b += term3 * term2_b;
//     return *this;
// }
// MySolver& MySolver::train_batch(COMP_T*   data,
//                                 DAT_DIM_T d,
//                                 N_DAT_T*  x,
//                                 N_DAT_T   n,
//                                 SUPV_T*   y,
//                                 COMP_T    eta) {
//     COMP_T term1 = 1 - eta * my_param->regul_coef();
//     for (DAT_DIM_T i = 0; i < d; ++i) {
//         term2[i] = 0;
//     }
//     COMP_T term2_b = 0;
//     COMP_T term3   = eta / n;
//     for (N_DAT_T i = 0; i < n; ++i) {
//         N_DAT_T x_i   = x[i];
//         COMP_T* dat_i = data + d * x_i;
//         COMP_T  score = compute_score(dat_i, d);
//         COMP_T  p_i;
//         if (score < -1) p_i = p_margin_neg[stat.index_of_label(y[x_i])];
//         else if (score > 1) p_i = p_margin_pos[stat.index_of_label(y[x_i])];
//         else p_i = p_margin_mid[stat.index_of_label(y[x_i])];
//         for (DAT_DIM_T i = 0; i < d; ++i) {
//             term2[i] += p_i * dat_i[i];
//         }
//         term2_b += p_i;
//     }
//     for (DAT_DIM_T i = 0; i < d; ++i) {
//         w[i] = term1 * w[i] + term3 * term2[i];
//     }
//     b += term3 * term2_b;
//     return *this;
// }

MySolver& MySolver::train_batch(COMP_T*   data,
                                DAT_DIM_T d,
                                N_DAT_T   n,
                                SUPV_T*   y,
                                COMP_T    eta) {
    COMP_T term1 = 1 - eta * my_param->regul_coef();
    for (DAT_DIM_T i = 0; i < d; ++i) {
        term2[i] = 0;
    }
    COMP_T  term2_b = 0;
    COMP_T* dat_i   = data;
    for (N_DAT_T i = 0; i < n; ++i, dat_i += d) {
        COMP_T  score   = compute_score(dat_i, d);
        SUPV_T  i_label = stat.index_of_label(y[i]);
        COMP_T  p_i;
        if     (score < -1) p_i = p_margin_neg[i_label];
        else if (score > 1) p_i = p_margin_pos[i_label];
        else                p_i = p_margin_mid[i_label];
        COMP_T  coeff   = p_i / stat.num_of_samples_with_label(i_label);
        for (DAT_DIM_T i = 0; i < d; ++i) {
            term2[i] += coeff * dat_i[i];
        }
        term2_b += coeff;
    }
    for (DAT_DIM_T i = 0; i < d; ++i) {
        w[i] = term1 * w[i] + eta * term2[i];
    }
    b += eta * term2_b;
    return *this;
}
MySolver& MySolver::train_batch(COMP_T*   data,
                                DAT_DIM_T d,
                                N_DAT_T*  x,
                                N_DAT_T   n,
                                SUPV_T*   y,
                                COMP_T    eta) {
    COMP_T term1 = 1 - eta * my_param->regul_coef();
    for (DAT_DIM_T i = 0; i < d; ++i) {
        term2[i] = 0;
    }
    COMP_T term2_b = 0;
    for (N_DAT_T i = 0; i < n; ++i) {
        N_DAT_T x_i     = x[i];
        COMP_T* dat_i   = data + d * x_i;
        COMP_T  score   = compute_score(dat_i, d);
        SUPV_T  i_label = stat.index_of_label(y[x_i]);
        COMP_T  p_i;
        if     (score < -1) p_i = p_margin_neg[i_label];
        else if (score > 1) p_i = p_margin_pos[i_label];
        else                p_i = p_margin_mid[i_label];
        COMP_T  coeff   = p_i / stat.num_of_samples_with_label(i_label);
        for (DAT_DIM_T i = 0; i < d; ++i) {
            term2[i] += coeff * dat_i[i];
        }
        term2_b += coeff;
    }
    for (DAT_DIM_T i = 0; i < d; ++i) {
        w[i] = term1 * w[i] + eta * term2[i];
    }
    b += eta * term2_b;
    return *this;
}

// Using Hinge Loss
// COMP_T MySolver::compute_obj(COMP_T*   data,
//                              DAT_DIM_T d,
//                              N_DAT_T   n,
//                              SUPV_T*   y) {
//     COMP_T loss  = 0;
//     COMP_T* dat_i = data;
//     for (N_DAT_T i = 0; i < n; ++i, dat_i += d) {
//         COMP_T  p_y      = p[stat.index_of_label(y[i])];
//         COMP_T  score    = compute_score(dat_i, d);
//         COMP_T  pos_loss = score < 1 ? 1 - score : 0;
//         COMP_T  neg_loss = score > -1 ? 1 + score : 0;
//         loss += p_y * pos_loss + (1 - p_y) * neg_loss;
//     }
//     return loss / n + 0.5 * my_param->regul_coef() * compute_norm(d);
// }
// // Using Hinge Loss
// COMP_T MySolver::compute_obj(COMP_T*   data,
//                              DAT_DIM_T d,
//                              N_DAT_T*  x,
//                              N_DAT_T   n,
//                              SUPV_T*   y) {
//     COMP_T loss = 0;
//     for (N_DAT_T i = 0; i < n; ++i) {
//         N_DAT_T x_i      = x[i];
//         COMP_T  p_y      = p[stat.index_of_label(y[x_i])];
//         COMP_T  score    = compute_score(data + d * x_i, d);
//         COMP_T  pos_loss = score < 1 ? 1 - score : 0;
//         COMP_T  neg_loss = score > -1 ? 1 + score : 0;
//         loss += p_y * pos_loss + (1 - p_y) * neg_loss;
//     }
//     return loss / n + 0.5 * my_param->regul_coef() * compute_norm(d);
// }

// Using Hinge Loss
COMP_T MySolver::compute_obj(COMP_T*   data,
                             DAT_DIM_T d,
                             N_DAT_T   n,
                             SUPV_T*   y) {
    COMP_T loss  = 0;
    SUPV_T n_label = stat.num_of_labels();
    for (SUPV_T k = 0; k < n_label; ++k) {
        N_DAT_T* x_label   = stat[k];
        N_DAT_T  n_x_label = stat.num_of_samples_with_label(k);
        COMP_T   p_k       = p[k];
        COMP_T   loss_k    = 0;
        for (N_DAT_T i = 0; i < n_x_label; ++i) {
            COMP_T  score    = compute_score(data + d * x_label[i], d);
            COMP_T  pos_loss = score < 1 ? 1 - score : 0;
            COMP_T  neg_loss = score > -1 ? 1 + score : 0;
            loss_k += p_k * pos_loss + (1 - p_k) * neg_loss;
        }
        loss += loss_k / n_x_label;
    }
    return loss + 0.5 * my_param->regul_coef() * compute_norm(d);
}
// Using Hinge Loss
COMP_T MySolver::compute_obj(COMP_T*   data,
                             DAT_DIM_T d,
                             N_DAT_T*  x,
                             N_DAT_T   n,
                             SUPV_T*   y) {
    COMP_T loss  = 0;
    SUPV_T n_label = stat.num_of_labels();
    for (SUPV_T k = 0; k < n_label; ++k) {
        N_DAT_T* x_label   = stat[k];
        N_DAT_T  n_x_label = stat.num_of_samples_with_label(k);
        COMP_T   p_k       = p[k];
        COMP_T   loss_k    = 0;
        for (N_DAT_T i = 0; i < n_x_label; ++i) {
            COMP_T  score    = compute_score(data + d * x_label[i], d);
            COMP_T  pos_loss = score < 1 ? 1 - score : 0;
            COMP_T  neg_loss = score > -1 ? 1 + score : 0;
            loss_k += p_k * pos_loss + (1 - p_k) * neg_loss;
        }
        loss += loss_k / n_x_label;
    }
    return loss + 0.5 * my_param->regul_coef() * compute_norm(d);
}

MySolver& MySolver::full_train(unsigned int& i,
                               unsigned int  n_iter,
                               DAT_DIM_T     dim,
                               N_DAT_T*      x,
                               N_DAT_T       n_sample,
                               SUPV_T        n_label) {
    char          gd_v        = gd_param->verbosity();
    bool   show_obj_each_iter = gd_param->show_obj_each_iteration();
    std::ostream* out         = my_param->ostream_of_training_process();
    char          my_v        = my_param->verbosity();
    COMP_T        err         = my_param->accuracy();
    bool          show_p      = my_param->show_p_each_iter();
    for (i = 0; i < n_iter; ++i) {
        if (my_v >= 2) {
            std::cout << "    Iteration " << i + 1 << " ... ";
            if (my_v >= 3) std::cout << "\n        Updating [w, b] ... ";
            if (gd_v >= 1) std::cout << std::endl;
            else std::cout.flush();
        }
        train();
        if (my_v >= 3) {
            if (gd_v < 1) std::cout << "Done.\n";
            std::cout << "        Updating p ... " << std::flush;
        }
        COMP_T p_diff = update_p(dim, n_label);
        if (my_v >= 2) {
            if (my_v == 2 && gd_v >= 1) std::cout << "\n        ";
            else std::cout << "Done. ";
            std::cout << "eta = " << eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = "
                          << compute_obj(data, dim, x, n_sample, y);
            }
            if (show_p) {
                std::cout << ", p = ";
                for (SUPV_T i = 0; i < n_label; ++i) {
                    std::cout << "[" << stat.label(i) << "|" << p[i] << "]";
                }
            }
            std::cout << "." << std::endl;
        }
        // DEBUG
        if (out) {
            this->ostream_param(*out);
            *out << std::endl;
        }
        if (err > 0 && p_diff < err) break;
    }
    return *this;
}

MySolver& MySolver::fine_train(unsigned int& i,
                               unsigned int  n_iter_fine,
                               DAT_DIM_T     dim,
                               N_DAT_T*      x,
                               N_DAT_T       n_sample,
                               SUPV_T        n_label) {
    char          gd_v        = gd_param->verbosity();
    bool   show_obj_each_iter = gd_param->show_obj_each_iteration();
    std::ostream* out         = my_param->ostream_of_training_process();
    char          my_v        = my_param->verbosity();
    COMP_T        err         = my_param->accuracy();
    bool          show_p      = my_param->show_p_each_iter();
    N_DAT_T       s_batch     = sgd_param->size_of_batches();
    N_DAT_T       n_batch     = n_sample / s_batch;
    N_DAT_T       n_remain    = n_sample - s_batch * n_batch;
    for (i = 0; i < n_iter_fine; ++i) {
        if (my_v >= 3)
            std::cout << "    Shuffling the data set... " << std::flush;
        rand_index(x, n_sample);
        if (my_v >= 3) std::cout << "Done." << std::endl; 
        if (my_v >= 2) {
            std::cout << "    Iteration " << i + 1 << " ... ";
            if (my_v >= 3) std::cout << "\n        Updating [w, b] ... ";
            if (gd_v >= 1) std::cout << std::endl;
            else std::cout.flush();
        }
        train_epoch(dim, x, s_batch, n_batch, n_remain, my_v);
        if (my_v >= 3) {
            if (gd_v < 1) std::cout << "Done.\n";
            std::cout << "        Updating p ... " << std::flush;
        }
        COMP_T p_diff = update_p(dim, n_label);
        if (my_v >= 2) {
            if (my_v == 2 && gd_v >= 1) std::cout << "\n        ";
            else std::cout << "Done. ";
            std::cout << "eta = " << eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = "
                          << compute_obj(data, dim, x, n_sample, y);
            }
            if (show_p) {
                std::cout << ", p = ";
                for (SUPV_T i = 0; i < n_label; ++i) {
                    std::cout << "[" << stat.label(i) << "|" << p[i] << "]";
                }
            }
            std::cout << "." << std::endl;
        }
        // DEBUG
        if (out) {
            this->ostream_param(*out);
            *out << std::endl;
        }
        if (err > 0 && p_diff < err) break;
    }
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

COMP_T MySolver::info_gain(N_DAT_T n_sample, SUPV_T n_label) const {
    if (!n_x_pos || !n_x_neg) return 0;
    COMP_T ent_pos = 0, ent_neg = 0;
    for (SUPV_T i = 0; i < n_label; ++i) {
        COMP_T p_i = p[i];
        COMP_T n_i = stat.num_of_samples_with_label(i);
        COMP_T p_k_pos = p_i * n_i / n_x_pos;
        COMP_T p_k_neg = (1 - p_i) * n_i / n_x_neg;
        ent_pos -= p_k_pos <= 0 ? 0 : p_k_pos * std::log2(p_k_pos);
        ent_neg -= p_k_neg <= 0 ? 0 : p_k_neg * std::log2(p_k_neg);
    }
    return stat.entropy() - (ent_pos * n_x_pos + ent_neg * n_x_neg) / n_sample;
}
