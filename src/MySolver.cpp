# include <iostream>
# include <fstream>
# include <iomanip>
# include <limits>
# include <cmath>
# include <functional>
# include <algorithm>
# include <vector>
# include <utility>

# include "MySolver.hpp"
# include "my_lib.hpp"

MySolver::MySolver():
          my_param(NULL),
          w(NULL),
          b(0),
          p(NULL),
          x_pos(NULL),
          n_x_pos(0),
          x_neg(NULL),
          n_x_neg(0),
          term_loss(NULL),
          supp_p(NULL),
          sort_min_max_p(true),
          sort_max_min_p(true),
          supp_p_min_max(-1),
          supp_p_max_min(2),
          supp_i_p_min_max(-1),
          supp_i_p_max_min(-1) {
}

MySolver::MySolver(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
                   SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
                   MyParam&  _my_param):
          SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>(_gd_param, _sgd_param),
          my_param(&_my_param),
          w(NULL),
          b(0),
          p(NULL),
          x_pos(NULL),
          n_x_pos(0),
          x_neg(NULL),
          n_x_neg(0),
          term_loss(NULL),
          supp_p(NULL),
          sort_min_max_p(true),
          sort_max_min_p(true),
          supp_p_min_max(-1),
          supp_p_max_min(2),
          supp_i_p_min_max(-1),
          supp_i_p_max_min(-1) {
}

MySolver::MySolver(MySolver& some):
          SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>(some),
          my_param(some.my_param),
          w(NULL),
          b(some.b),
          p(NULL),
          x_pos(some.x_pos),
          n_x_pos(some.n_x_pos),
          x_neg(some.x_neg),
          n_x_neg(some.n_x_neg),
          term_loss(some.term_loss),
          supp_p(NULL),
          sort_min_max_p(some.sort_min_max_p),
          sort_max_min_p(some.sort_max_min_p),
          supp_p_min_max(some.supp_p_min_max),
          supp_p_max_min(some.supp_p_max_min),
          supp_i_p_min_max(some.supp_i_p_min_max),
          supp_i_p_max_min(some.supp_i_p_max_min) {
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
    if (some.supp_p) {
        supp_p = new COMP_T[n_label];
        std::memcpy(supp_p, some.supp_p, n_label * sizeof(COMP_T));
    }
}

MySolver& MySolver::ostream_this(std::ostream& out) {
    stat.ostream_this(out);
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

MySolver& MySolver::ofstream_this(std::ofstream& out) {
    DAT_DIM_T dim = gd_param->dimension();
    out << stat;
    if (w) {
        out << dim << "    # feature dimension\n";
        for (DAT_DIM_T i = 0; i < dim; ++i) {
            out << std::setprecision(15) << w[i] << "    # w[" << i << "]\n";
        }
        out << std::setprecision(15) << b << "    # b\n";
    }
    else {
        out << "0    # feature dimension\n";
    }
    return *this;
}

MySolver& MySolver::istream_this(std::istream& in) {
    clear();
    in >> stat;
    char line_str[SIZEOF_LINE];
    in.getline(line_str, SIZEOF_LINE);
    DAT_DIM_T dim = strto<DAT_DIM_T>(line_str);
    if (dim) {
        w = new COMP_T[dim];
        for (DAT_DIM_T i = 0; i < dim; ++i) {
            in.getline(line_str, SIZEOF_LINE);
            w[i] = strto<COMP_T>(line_str);
        }
        in.getline(line_str, SIZEOF_LINE);
        b = strto<COMP_T>(line_str);
    }
    return *this;
}

MySolver& MySolver::solve() {
    DAT_DIM_T    dim               = gd_param->dimension();
    char         gd_v              = gd_param->verbosity();
    N_DAT_T      n_subsample       = gd_param->min_num_of_subsamples();
    COMP_T       eta0              = gd_param->init_learning_rate();
    bool         show_obj          = gd_param->show_obj_each_iteration();
    char         my_v              = my_param->verbosity();
    unsigned int n_trial           = my_param->num_of_trials();
    unsigned int n_train           = my_param->num_of_trainings();
    unsigned int n_iter            = my_param->num_of_iterations();
    unsigned int n_iter_fine       = my_param->num_of_fine_tuning();
    float        supp_p_ratio      = my_param->support_p_ratio();
    float        init_supp_p_ratio = my_param->initial_support_p_ratio();
    float        supp_p_inc_intv   = my_param->support_p_incre_interval();
    SUPV_T       n_inc_supp_p      = my_param->num_of_incre_support_ps();
    COMP_T       err               = my_param->accuracy();
    bool         show_p            = my_param->show_p_each_iter();
    SUPV_T       n_label           = stat.num_of_labels();
    N_DAT_T      n_sample          = stat.num_of_samples();
    N_DAT_T*     x                 = stat.index_of_samples();
    SUPV_T       n_supp_p      = supp_p_ratio * n_label + 0.5;
    SUPV_T       n_init_supp_p = init_supp_p_ratio * n_supp_p + 0.5;
    COMP_T       loss_opt      = std::numeric_limits<COMP_T>::max();
    MySolver*    best_solver   = NULL;
    MySolver*    try_solver    = NULL;
    COMP_T*      center        = new COMP_T[dim];
    if (!w) w = new COMP_T[dim];
    if (!p) p = new COMP_T[n_label];
    term_loss = new COMP_T[dim];
    supp_p    = new COMP_T[n_label];

    if (my_v >= 1) {
        std::cout << "MySolver Training: \n    Data: " << n_sample << " samples, "
                  << dim << " feature dimensions.\n    Stopping Criterion: "
                  << n_iter << " full iterations and " << n_iter_fine 
                  << " fine-tuning iterations";
        if (err > 0) std::cout << " or accuracy higher than " << err;
        std::cout <<  ".\nMySolver Training: begin." << std::endl;
    }
    
    if (!n_subsample || n_subsample > n_sample) n_subsample = n_sample;

    if (!n_supp_p && supp_p_ratio > 0) n_supp_p = 1;
    if (!n_init_supp_p && init_supp_p_ratio > 0) n_init_supp_p = 1;
    if (n_supp_p >= n_label) n_supp_p = n_label;
    if (n_supp_p > 0) {
        SUPV_T n_supp_p_rest = n_supp_p - n_init_supp_p;
        if (n_supp_p_rest > 0) {
            if (n_inc_supp_p > n_supp_p_rest) n_inc_supp_p = n_supp_p_rest;
            if (supp_p_inc_intv > 0) {
                if (!n_inc_supp_p) {
                    n_inc_supp_p = 1 + (n_supp_p_rest - 1) / (unsigned int)(n_iter / supp_p_inc_intv);
                }
            }
            else {
                if (n_inc_supp_p) {
                    supp_p_inc_intv = (float)n_iter / (1 + (n_supp_p_rest - 1) / n_inc_supp_p);
                    if (supp_p_inc_intv < 1) {
                        n_inc_supp_p = 1 + (n_supp_p_rest - 1) / n_iter;
                        supp_p_inc_intv = 1;
                    }
                }
                else {
                    if (n_iter >= (unsigned)n_supp_p_rest) {
                        n_inc_supp_p = 1;
                        supp_p_inc_intv = (float)n_iter / n_supp_p_rest;
                    }
                    else {
                        n_inc_supp_p = 1 + (n_supp_p_rest - 1) / n_iter;
                        supp_p_inc_intv = (float)n_iter / (1 + (n_supp_p_rest - 1) / n_inc_supp_p);
                    }
                }
            }
        }
    }

    t = 0;
    for (unsigned int i_trial = 0; i_trial < n_trial; ++i_trial) {
        if (my_v >= 1) {
            std::cout << "Trying parameters " << i_trial + 1 << " ... \n"
                      << "Initializing ... " << std::flush;
        }
        rand_index(x, n_sample);
        for (DAT_DIM_T i = 0; i < dim; ++i) center[i] = 0;
        for (N_DAT_T i = 0; i < n_subsample; ++i) {
            COMP_T* dat_i = data + dim * x[i];
            for (DAT_DIM_T j = 0; j < dim; ++j) center[j] += dat_i[j];
        }
        for (DAT_DIM_T i = 0; i < dim; ++i) center[i] /= n_subsample;

        COMP_T* sample = data + dim * x[n_sample - 1];
        b = 0;
        for (DAT_DIM_T i = 0; i < dim; ++i) {
            w[i] = sample[i] - center[i];
            b -= w[i] * center[i];
        }
        for (SUPV_T i = 0; i < n_label; ++i) p[i] = 0;
        reset_support_p(n_label);
        COMP_T p_mean = update_p(dim, n_label);
        SUPV_T n_cur_supp_p;
        if (p_mean > 0.5) n_cur_supp_p = add_support_p(-n_init_supp_p, n_label);
        else n_cur_supp_p = add_support_p(n_init_supp_p, n_label);
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
            // unsigned int i, j;
            if (try_solver) try_solver->assign_to_this(this);
            else try_solver = new MySolver(*this);
            try_solver->full_train(n_iter, dim, x, n_sample, n_label,
                                   eta0, n_supp_p, n_cur_supp_p, n_inc_supp_p,
                                   supp_p_inc_intv, p_mean);
            if (my_v == 1 && gd_v < 1) std::cout << "Done." << std::endl;
            // if (my_v >= 1) {
            //     if (my_v > 1 || gd_v >= 1) std::cout << std::endl;
            //     else std::cout.flush();
            // }
            if (n_iter_fine > 0) {
                if (my_v >= 1) {
                    std::cout << "Fine-tuning ... ";
                    if (my_v > 1 || gd_v >= 1) std::cout << std::endl;
                    else std::cout.flush();
                }
                try_solver->fine_train(n_iter_fine, dim, x, n_sample, n_label);
                if (my_v == 1 && gd_v < 1) std::cout << "Done." << std::endl;
            }
            if (try_solver->fail()) {
                if (my_v >= 1) {
                    std::cout << "Normal training gives trivial solutions. Simple training ... ";
                    if (my_v > 1 || gd_v >= 1) std::cout << std::endl;
                    else std::cout.flush();
                }
                try_solver->assign_to_this(this);
                try_solver->simple_train(dim, n_label);
                if (my_v == 1 && gd_v < 1) std::cout << "Done." << std::endl;
            }
            if (my_v >= 1) {
                std::cout << "Training " << i_train + 1 << ": finished.";
                // if (i < n_iter)
                //     std::cout << "    Training stopped at iteration " << i + 1 
                //               << " with convergence.";
                // else if (j < n_iter_fine) {
                //     std::cout << "    Training stopped during fine-tuning iteration "
                //               << j + 1 << " with convergence.";
                // }
                // else std::cout << "    Max number of iterations has been reached.";
                std::cout << std::endl;
            }
            COMP_T loss = try_solver->compute_loss(data, dim, x, n_sample, y);
            if (loss < loss_opt) {
                MySolver* temp = try_solver;
                try_solver  = best_solver;
                best_solver = temp;
                if (my_v >= 1) {
                    std::cout << "    Current result is the best so far.";
                    if (show_obj) {
                        std::cout << " (Loss: Old = " << loss_opt << ", New = "
                                  << loss << ")";
                    }
                    std::cout << std::endl;
                }
                loss_opt = loss;
            }
            else {
                if (my_v >= 1) {
                    std::cout << "    Current result is not the best.";
                    if (show_obj) {
                        std::cout << " (Loss: Best = " << loss_opt << ", New = "
                                  << loss << ")";
                    }
                    std::cout << std::endl;
                }
            }
        }
    }
    if (best_solver) {
        assign_to_this(best_solver);
        if (x_pos && x_neg) update_p(dim, n_label);
    }
    delete   best_solver;
    delete   try_solver;
    delete[] term_loss;    term_loss    = NULL;
    delete[] supp_p;       supp_p       = NULL;
    delete[] center;
    if (my_v >= 1) {
        std::cout << "MySolver Training: finished. \n";
    }
    return *this;
}

COMP_T MySolver::update_p(DAT_DIM_T d, SUPV_T n_label) {
    COMP_T diff_p  = 0;
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
                if (x_pos) x_pos[n_x_pos] = x_i;
                ++n_x_pos;
            }
            else {
                if (x_neg) x_neg[n_x_neg] = x_i;
                ++n_x_neg;
            }
        }
        COMP_T p_new = (COMP_T)n_x_label_pos / n_x_label;
        COMP_T p_old = p[i];
        COMP_T supp_p_i = supp_p[i];
        if (supp_p_i >= 0) {
            p_old = supp_p_i;
            supp_p_i = p[i];
            if (supp_p_i < 0.5) {
                if (p_new < supp_p_i) {
                    supp_p[i] = p_new;
                    if (supp_i_p_min_max == i) {
                        sort_min_max_p = false;
                    }
                    else if (supp_i_p_max_min == i) {
                        sort_max_min_p = false;
                    }
                }
                else {
                    supp_p[i] = p[i];
                }
            }
            else {
                if (p_new > supp_p_i) {
                    supp_p[i] = p_new;
                    if (supp_i_p_max_min == i) {
                        sort_max_min_p = false;
                    }
                    else if (supp_i_p_min_max == i) {
                        sort_min_max_p = false;
                    }
                }
                else {
                    supp_p[i] = p[i];
                }
            }
        }
        diff_p += p_new - p_old;
        p[i]    = p_new;
    }
    for (SUPV_T i = 0; i < n_label; ++i) {
        if (supp_p[i] >= 0) continue;
        COMP_T p_i = p[i];
        if (p_i < 0.5) {
            if (!sort_min_max_p) {
                supp_p_min_max = -1;
                for (SUPV_T i = 0; i < n_label; ++i) {
                    if (supp_p[i] >= 0) {
                        if (supp_p[i] > supp_p_min_max) {
                            supp_p_min_max   = supp_p[i];
                            supp_i_p_min_max = i;
                        }
                    }
                }
                sort_min_max_p = true;
            }
            if (supp_p_min_max > p_i) {
                supp_p[supp_i_p_min_max] = -1;
                supp_p[i]                = p_i;
                sort_min_max_p           = false;
            }
        }
        else {
            if (!sort_max_min_p) {
                supp_p_max_min = 2;
                for (SUPV_T i = 0; i < n_label; ++i) {
                    if (supp_p[i] > 0.5) {
                        if (supp_p[i] < supp_p_max_min) {
                            supp_p_max_min   = supp_p[i];
                            supp_i_p_max_min = i;
                        }
                    }
                }
                sort_max_min_p = true;
            }
            if (supp_p_max_min < p_i) {
                supp_p[supp_i_p_max_min] = -1;
                supp_p[i]                = p_i;
                sort_max_min_p           = false;
            }
        }
    }
    return (COMP_T)n_x_pos / (n_x_pos + n_x_neg);
}

MySolver& MySolver::reset_support_p(SUPV_T n_label) {
    for (SUPV_T i = 0; i < n_label; ++i) supp_p[i] = -1;
    sort_max_min_p = true;
    sort_min_max_p = true;
    supp_p_min_max = -1;
    supp_p_max_min = 2;
    return *this;
}

MySolver& MySolver::support_p(SUPV_T n_label) {
    COMP_T n_pos = 0;
    for (SUPV_T i = 0; i < n_label; ++i) {
        COMP_T supp_p_i = supp_p[i];
        if (supp_p_i >= 0) {
            supp_p[i] = p[i];
            p[i] = supp_p_i;
        }
        n_pos += p[i] * stat.num_of_samples_with_label(i);
    }
    n_x_pos = n_pos;
    n_x_neg = stat.num_of_samples() - n_x_pos;
    return *this;
}

SUPV_T MySolver::add_support_p(SUPV_T n_add_supp_p, SUPV_T n_label) {
    COMP_T cur_p_m;
    SUPV_T cur_i_p_m;
    if (n_add_supp_p > 0) {
        for (SUPV_T i = 0; i < n_add_supp_p; ++i) {
            cur_p_m = -1;
            for (SUPV_T j = 0; j < n_label; ++j) {
                if (supp_p[j] >= 0) continue;
                COMP_T p_j = p[j];
                if (p_j > cur_p_m) {
                    cur_p_m   = p_j;
                    cur_i_p_m = j;
                }
            }
            if (cur_p_m < 0.5) return i;
            supp_p[cur_i_p_m] = cur_p_m;
            if (sort_max_min_p) {
                if (cur_p_m < supp_p_max_min) {
                    supp_p_max_min = cur_p_m;
                    supp_i_p_max_min = cur_i_p_m;
                }
            }
        }

    }
    else if (n_add_supp_p < 0) {
        for (SUPV_T i = 0; i > n_add_supp_p; --i) {
            cur_p_m = 2;
            for (SUPV_T j = 0; j < n_label; ++j) {
                if (supp_p[j] >= 0 && supp_p[j] <= 1) continue;
                COMP_T p_j = p[j];
                if (p_j < cur_p_m) {
                    cur_p_m   = p_j;
                    cur_i_p_m = j;
                }
            }
            if (cur_p_m > 0.5) return -i;
            supp_p[cur_i_p_m] = cur_p_m;
            if (sort_min_max_p) {
                if (cur_p_m > supp_p_min_max) {
                    supp_p_min_max = cur_p_m;
                    supp_i_p_min_max = cur_i_p_m;
                }
            }
        }
    }
    return n_add_supp_p >= 0 ? n_add_supp_p : -n_add_supp_p;
}

// AVERAGE THE WHOLE LOSS
// MySolver& MySolver::train_batch(COMP_T*   data,
//                                 DAT_DIM_T d,
//                                 N_DAT_T*  x,
//                                 N_DAT_T   n,
//                                 SUPV_T*   y,
//                                 COMP_T    eta) {
//     COMP_T term1 = 1 - eta * my_param->regul_coef();
//     for (DAT_DIM_T i = 0; i < d; ++i) {
//         term_loss[i] = 0;
//     }
//     COMP_T term2_b = 0;
//     COMP_T term3   = eta / n;
//     for (N_DAT_T i = 0; i < n; ++i) {
//         N_DAT_T x_i   = x[i];
//         COMP_T* dat_i = data + d * x_i;
//         COMP_T  score = compute_score(dat_i, d);
//         COMP_T  p_i;
//         if (score < -1) p_i = p[stat.index_of_label(y[x_i])];
//         else if (score > 1) p_i = p[stat.index_of_label(y[x_i])] - 1;
//         else p_i = 2 * p[stat.index_of_label(y[x_i])] - 1;
//         for (DAT_DIM_T i = 0; i < d; ++i) {
//             term_loss[i] += p_i * dat_i[i];
//         }
//         term2_b += p_i;
//     }
//     for (DAT_DIM_T i = 0; i < d; ++i) {
//         w[i] = term1 * w[i] + term3 * term_loss[i];
//     }
//     b += my_param->bias_learning_rate_factor() * term3 * term2_b;
//     return *this;
// }

// AVERAGE LOSS OF POSITIVE AND NEGATIVE SAMPLES
MySolver& MySolver::train_batch(COMP_T*   data,
                                DAT_DIM_T d,
                                N_DAT_T*  x,
                                N_DAT_T   n,
                                SUPV_T*   y,
                                COMP_T    eta) {
    COMP_T reg_w = 1 - eta * my_param->regul_coef();
    for (DAT_DIM_T i = 0; i < d; ++i) term_loss[i] = 0;
    COMP_T term_loss_b = 0;
    COMP_T coeff_pos = 1 / (COMP_T)n_x_pos;
    COMP_T coeff_neg = 1 / (COMP_T)n_x_neg;
    for (N_DAT_T i = 0; i < n; ++i) {
        N_DAT_T x_i   = x[i];
        COMP_T* dat_i = data + d * x_i;
        COMP_T  score = compute_score(dat_i, d);
        COMP_T  coeff;
        if (score < -1) 
            coeff = p[stat.index_of_label(y[x_i])] * coeff_pos;
        else if (score > 1) 
            coeff = (p[stat.index_of_label(y[x_i])] - 1) * coeff_neg;
        else 
            coeff = p[stat.index_of_label(y[x_i])] * (coeff_neg + coeff_pos) - coeff_neg;
        for (DAT_DIM_T i = 0; i < d; ++i) {
            term_loss[i] += coeff * dat_i[i];
        }
        term_loss_b += coeff;
    }

    // std::cout << "\nGradient check: \n";
    // COMP_T obj = compute_obj(data, d, x, n, y);
    // w[0] += 1e-4;
    // std::cout << term_loss[0] - my_param->regul_coef() * w[0]
    //           << " | " << (compute_obj(data, d, x, n, y) - obj) / 1e-4 << "\n";
    // for (DAT_DIM_T i = 1; i < d; ++i) {
    //     w[i - 1] -= 1e-4;
    //     w[i] += 1e-4;
    //     std::cout << term_loss[i] - my_param->regul_coef() * w[i]
    //               << " | " << (compute_obj(data, d, x, n, y) - obj) / 1e-4 << "\n";
    // }
    // w[d - 1] -= 1e-4;
    // b += 1e-4;
    // std::cout << term_loss_b << " | " << (compute_obj(data, d, x, n, y) - obj) / 1e-4;
    // b -= 1e-4;
    // std::cout << std::endl;

    for (DAT_DIM_T i = 0; i < d; ++i) {
        w[i] = reg_w * w[i] + eta * term_loss[i];
    }
    if (my_param->regul_bias()) b = reg_w * b;
    b += my_param->bias_learning_rate_factor() * eta * term_loss_b;
    return *this;
}

// AVERAGE LOSS OF EVERY LABEL OF SAMPLES
// MySolver& MySolver::train_batch(COMP_T*   data,
//                                 DAT_DIM_T d,
//                                 N_DAT_T*  x,
//                                 N_DAT_T   n,
//                                 SUPV_T*   y,
//                                 COMP_T    eta) {
//     COMP_T term1 = 1 - eta * my_param->regul_coef();
//     for (DAT_DIM_T i = 0; i < d; ++i) {
//         term_loss[i] = 0;
//     }
//     COMP_T term2_b = 0;
//     for (N_DAT_T i = 0; i < n; ++i) {
//         N_DAT_T x_i     = x[i];
//         COMP_T* dat_i   = data + d * x_i;
//         COMP_T  score   = compute_score(dat_i, d);
//         SUPV_T  i_label = stat.index_of_label(y[x_i]);
//         COMP_T  p_i;
//         if     (score < -1) p_i = p_margin_neg[i_label];
//         else if (score > 1) p_i = p_margin_pos[i_label];
//         else                p_i = p_margin_mid[i_label];
//         COMP_T  coeff   = p_i / stat.num_of_samples_with_label(i_label);
//         for (DAT_DIM_T i = 0; i < d; ++i) {
//             term_loss[i] += coeff * dat_i[i];
//         }
//         term2_b += coeff;
//     }
//     for (DAT_DIM_T i = 0; i < d; ++i) {
//         w[i] = term1 * w[i] + eta * term_loss[i];
//     }
//     b += eta * term2_b;
//     return *this;
// }

COMP_T MySolver::compute_obj(COMP_T*   data,
                             DAT_DIM_T d,
                             N_DAT_T*  x,
                             N_DAT_T   n,
                             SUPV_T*   y) {
    COMP_T loss = compute_loss(data, d, x, n, y);
    COMP_T reg_b = 0;
    if (my_param->regul_bias()) reg_b = b * b;
    return loss + 0.5 * my_param->regul_coef() * (compute_norm(d) + reg_b);
}


// AVERAGE THE WHOLE LOSS
// Using Hinge Loss
// COMP_T MySolver::compute_loss(COMP_T*   data,
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
//     return loss / n;
// }
// // Using Hinge Loss
// COMP_T MySolver::compute_loss(COMP_T*   data,
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
//     return loss / n;
// }


// AVERAGE THE LOSS OF POSITIVE AND NEGATIVE SAMPLES
// Using Hinge Loss
COMP_T MySolver::compute_loss(COMP_T*   data,
                              DAT_DIM_T d,
                              N_DAT_T*  x,
                              N_DAT_T   n,
                              SUPV_T*   y) {
    COMP_T loss = 0;
    COMP_T coeff_pos = 1 / (COMP_T)n_x_pos;
    COMP_T coeff_neg = 1 / (COMP_T)n_x_neg;
    for (N_DAT_T i = 0; i < n; ++i) {
        N_DAT_T x_i      = x[i];
        COMP_T  p_y      = p[stat.index_of_label(y[x_i])];
        COMP_T  score    = compute_score(data + d * x_i, d);
        COMP_T  pos_loss = score < 1 ? 1 - score : 0;
        COMP_T  neg_loss = score > -1 ? 1 + score : 0;
        loss += p_y * pos_loss * coeff_pos + (1 - p_y) * neg_loss * coeff_neg;
    }
    return loss;
}

// AVERAGE LOSS OF EVERY LABEL OF SAMPLES
// Using Hinge Loss
// COMP_T MySolver::compute_loss(COMP_T*   data,
//                              DAT_DIM_T d,
//                              N_DAT_T*  x,
//                              N_DAT_T   n,
//                              SUPV_T*   y) {
//     COMP_T loss  = 0;
//     SUPV_T n_label = stat.num_of_labels();
//     for (SUPV_T k = 0; k < n_label; ++k) {
//         N_DAT_T* x_label   = stat[k];
//         N_DAT_T  n_x_label = stat.num_of_samples_with_label(k);
//         COMP_T   p_k       = p[k];
//         COMP_T   loss_k    = 0;
//         for (N_DAT_T i = 0; i < n_x_label; ++i) {
//             COMP_T  score    = compute_score(data + d * x_label[i], d);
//             COMP_T  pos_loss = score < 1 ? 1 - score : 0;
//             COMP_T  neg_loss = score > -1 ? 1 + score : 0;
//             loss_k += p_k * pos_loss + (1 - p_k) * neg_loss;
//         }
//         loss += loss_k / n_x_label;
//     }
//     return loss;
// }

MySolver& MySolver::full_train(unsigned int  n_iter,
                               DAT_DIM_T     dim,
                               N_DAT_T*      x,
                               N_DAT_T       n_sample,
                               SUPV_T        n_label,
                               COMP_T        eta0,
                               SUPV_T        n_supp_p,
                               SUPV_T        cur_n_supp_p,
                               SUPV_T        n_inc_supp_p,
                               float         supp_p_inc_intv,
                               COMP_T        p_mean) {
    char          gd_v            = gd_param->verbosity();
    bool       show_obj_each_iter = gd_param->show_obj_each_iteration();
    std::ostream* out             = my_param->ostream_of_training_process();
    char          my_v            = my_param->verbosity();
    bool          show_p          = my_param->show_p_each_iter();
    float         supp_p_inc_step = 0;
    COMP_T        p_mean_old      = 0.5;
    for (unsigned int i = 0; i < n_iter; ++i) {
        if (my_v >= 2) {
            std::cout << "    Iteration " << i + 1 << " ... ";
            if (my_v >= 3) std::cout << "\n        Supporting p ... " << std::flush;
        }
        support_p(n_label);
        if (cur_n_supp_p < n_supp_p) {
            supp_p_inc_step += 1 / supp_p_inc_intv;
            if (supp_p_inc_step >= 1) {
                SUPV_T inc_supp_p;
                if (p_mean - p_mean_old > 0) inc_supp_p = add_support_p(-n_inc_supp_p, n_label);
                else inc_supp_p = add_support_p(n_inc_supp_p, n_label);
                supp_p_inc_step -= 1;
                cur_n_supp_p    += inc_supp_p;
                p_mean_old = p_mean;
            }   
        }
        if (my_v >= 3) {
            std::cout << "Done. ";
            if (show_p) {
                std::cout << cur_n_supp_p << " supporting ps, \n            p = ";
                for (SUPV_T i = 0; i < n_label; ++i) {
                    if (supp_p[i] >= 0 && supp_p[i] <= 1)
                        std::cout << "{" << stat.label(i) << "|" << p[i] << "}";
                    else
                        std::cout << "[" << stat.label(i) << "|" << p[i] << "]";
                }
            }
            std::cout << "\n        Updating [w, b] ... ";
            if (gd_v >= 1) std::cout << std::endl;
            else std::cout.flush();
        }
        gd_param->learning_rate_1st_try(eta);
        gd_param->init_learning_rate(eta0);
        train();
        if (my_v >= 3) {
            if (gd_v < 1) std::cout << "Done.\n";
            std::cout << "        Updating p ... " << std::flush;
        }
        p_mean = update_p(dim, n_label);
        if (my_v >= 2) {
            if (my_v == 2 && gd_v >= 1) std::cout << "\n        ";
            else std::cout << "Done. ";
            std::cout << "eta = " << eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = "
                          << compute_obj(data, dim, x, n_sample, y);
            }
            if (show_p) {
                std::cout << ",\n            p = ";
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
        // if (err > 0 && p_diff < err) break;
    }
    return *this;
}

MySolver& MySolver::fine_train(unsigned int  n_iter_fine,
                               DAT_DIM_T     dim,
                               N_DAT_T*      x,
                               N_DAT_T       n_sample,
                               SUPV_T        n_label) {
    char          gd_v        = gd_param->verbosity();
    bool   show_obj_each_iter = gd_param->show_obj_each_iteration();
    std::ostream* out         = my_param->ostream_of_training_process();
    char          my_v        = my_param->verbosity();
    // COMP_T        err         = my_param->accuracy();
    bool          show_p      = my_param->show_p_each_iter();
    N_DAT_T       s_batch     = sgd_param->size_of_batches();
    N_DAT_T       n_batch     = n_sample / s_batch;
    N_DAT_T       n_remain    = n_sample - s_batch * n_batch;
    for (unsigned int i = 0; i < n_iter_fine; ++i) {
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
        support_p(n_label);
        train_epoch(dim, x, s_batch, n_batch, n_remain, my_v);
        if (my_v >= 3) {
            if (gd_v < 1) std::cout << "Done.\n";
            std::cout << "        Updating p ... " << std::flush;
        }
        update_p(dim, n_label);
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
        // if (err > 0 && p_diff < err) break;
    }
    return *this;
}

MySolver& MySolver::simple_train(DAT_DIM_T dim, SUPV_T n_label) {
    std::ostream* out = my_param->ostream_of_training_process();
    SUPV_T  first, second;
    N_DAT_T n_first = 0, n_second = 0;
    for (SUPV_T i = 0; i < n_label; ++i) {
        N_DAT_T n_x_label = stat.num_of_samples_with_label(i);
        if (n_x_label > n_first) {
            n_first = n_x_label;
            first   = i;
        }
    }
    for (SUPV_T i = 0; i < n_label; ++i) {
        if (i == first) continue;
        N_DAT_T n_x_label = stat.num_of_samples_with_label(i);
        if (n_x_label > n_second) {
            n_second = n_x_label;
            second   = i;
        }
    }
    n_x_pos = n_first;
    n_x_neg = n_second;
    for (SUPV_T i = 0; i < n_label; ++i) p[i] = 0.5;
    p[first] = 1;
    p[second] = 0;
    train();
    update_p(dim, n_label);
    if (out) {
        this->ostream_param(*out);
        *out << std::endl;
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
