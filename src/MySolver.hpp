# ifndef _MYSOLVER_HPP
# define _MYSOLVER_HPP

# include <vector>
# include <utility>

# include "my_typedefs.h"
# include "SGD.hpp"


/// MySolver class
/// 
/// Dedicated to find a hyperplane in the feature space that well separate the 
/// samples by their labels.
class MySolver : public SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> {
public:
    class MyParam {
    public:
        MyParam(char         _v                = 1,
                COMP_T       _lambda           = 0.5,
                unsigned int _n_trial          = 5,
                unsigned int _n_train          = 1,
                unsigned int _n_iter           = 50,
                unsigned int _n_iter_fine      = 50,
                SUPV_T       _n_supp_p         = 1,
                float        _supp_p_inc_rate  = 1,
                COMP_T       _err              = 0.01,
                bool         _show_p_each_iter = false)
               :v(_v),
                lambda(_lambda),
                n_trial(_n_trial),
                n_train(_n_train),
                n_iter(_n_iter),
                n_iter_fine(_n_iter_fine),
                n_supp_p(_n_supp_p),
                supp_p_inc_rate(_supp_p_inc_rate),
                err(_err),
                show_p_each_iter_(_show_p_each_iter),
                out_training_proc(NULL) {}
        ~MyParam() {}
        
        MyParam& verbosity(char _v) { v = _v; return *this; }
        MyParam& regul_coef(COMP_T _lambda) { lambda = _lambda; return *this; }
        MyParam& num_of_trials(unsigned int _n_trial)
            { n_trial = _n_trial; return *this; }
        MyParam& num_of_trainings(unsigned int _n_train)
            { n_train = _n_train; return *this; }
        MyParam& num_of_iterations(unsigned int _n_iter)
            { n_iter = _n_iter; return *this; }
        MyParam& num_of_fine_tuning(unsigned int _n_iter_fine)
            { n_iter_fine = _n_iter_fine; return *this; }
        MyParam& num_of_support_ps(unsigned int _n_supp_p)
            { n_supp_p = _n_supp_p; return *this; }
        MyParam& support_p_incre_rate(float _supp_p_inc_rate)
            { supp_p_inc_rate = _supp_p_inc_rate; return *this; }
        MyParam& accuracy(COMP_T _err) { err = _err; return *this; }
        MyParam& show_p_each_iter(bool _show)
            { show_p_each_iter_ = _show; return *this; }
        MyParam& ostream_of_training_process(std::ostream& _out_training_proc)
            { out_training_proc = &_out_training_proc; return *this; }

        char          verbosity()                   const { return v; }
        COMP_T        regul_coef()                  const { return lambda; }
        unsigned int  num_of_trials()               const { return n_trial; }
        unsigned int  num_of_trainings()            const { return n_train; }
        unsigned int  num_of_iterations()           const { return n_iter; }
        unsigned int  num_of_fine_tuning()          const { return n_iter_fine; }
        SUPV_T        num_of_support_ps()           const { return n_supp_p; }
        float         support_p_incre_rate()        const { return supp_p_inc_rate; }
        COMP_T        accuracy()                    const { return err; }
        bool          show_p_each_iter()            const { return show_p_each_iter_; }
        std::ostream* ostream_of_training_process() const { return out_training_proc; }

        MyParam& ostream_this(std::ostream& out) {
            out << "MySolver parameters:\n"
                << "    Verbosity = " << (int)v << "\n"
                << "    Regularization coefficient = " << lambda << "\n"
                << "    Number of initial guesses = " << n_trial << "\n"
                << "    Number of trainings for each guess = " << n_train << "\n"
                << "    Max number of full iterations = " << n_iter << "\n"
                << "    Stopping accuracy = " << err;
            return *this;
        }

    private:
        char          v;
        COMP_T        lambda;      ///< trade-off between regularization & loss
        unsigned int  n_trial;     ///< number of intial guesses for w & b
        unsigned int  n_train;     ///< number of times of training
        unsigned int  n_iter;      ///< number of normal iterations
        unsigned int  n_iter_fine; ///< number of extra iterations for fine tuning
        SUPV_T        n_supp_p;    ///< number of support "p"s
        float         supp_p_inc_rate;
        COMP_T        err;
        bool          show_p_each_iter_;
        std::ostream* out_training_proc;
    };//class MyParam

    MySolver(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
             SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
             char         _v                = 1,
             COMP_T       _lambda           = 0.5,
             unsigned int _n_trial          = 5,
             unsigned int _n_train          = 1,
             unsigned int _n_iter           = 50,
             unsigned int _n_iter_fine      = 50,
             SUPV_T       _n_supp_p         = 1,
             float        _supp_p_inc_rate  = 1,
             COMP_T       _err              = 0.01,
             bool         _show_p_each_iter = false);
    MySolver(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
             SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
             MyParam&  _my_param);
    MySolver(MySolver& some);
    virtual ~MySolver();

    // Inherited functions
    virtual MySolver& ostream_this(std::ostream& out);
    virtual MySolver& ostream_param(std::ostream& out);

    /// The general solver of the class MySolver.
    /// 
    /// According to the training data "dat", "n", and "y", train the parameters 
    /// "w" and "b".
    /// @param[out] x_pos   An array of indexes of samples on the positive side 
    ///                     of the hyperplane.
    /// @param[out] n_x_pos The number of samples in "x_pos".
    /// @param[out] x_neg   An array of indexes of samples on the negative side 
    ///                     of the hyperplane.
    /// @param[out] n_x_neg The number of samples in "x_neg".
    /// @warning    "x_pos" & "x_neg" should point to pre-allocated memory spaces 
    ///             of at least the equal size of "stat.num_of_samples()".
    MySolver&         solve(N_DAT_T* _x_pos, N_DAT_T& _n_x_pos,
                            N_DAT_T* _x_neg, N_DAT_T& _n_x_neg);
    MySolver&         solve();
    COMP_T            entropy();
    N_DAT_T           num_of_samples() const { return stat.num_of_samples(); }
    LabelStat<SUPV_T, N_DAT_T>& 
                      distribution() { return stat; }

    /// Test one sample to judge whether the sample lies in the positive side or 
    /// negative side of the hyperplane "w'x+b=0".
    /// 
    /// @return     0 if positive, 1 if negative.
    virtual SUPV_T    test_one(COMP_T* dat_i, DAT_DIM_T d) const;

protected:
    /// Update the information of "p" and internal buffer "p_margin_...".
    /// 
    /// @return     The amount of difference of "p".
    COMP_T            update_p(DAT_DIM_T d, SUPV_T n_label);
    MySolver&         reset_support_p();
    MySolver&         support_p();
    SUPV_T            add_support_p(SUPV_T n_add_supp_p, SUPV_T n_label);

    // Inherited functions
    virtual COMP_T    compute_learning_rate(COMP_T eta0, unsigned long t);
    virtual MySolver& train_batch(COMP_T*   data,
                                  DAT_DIM_T d,
                                  N_DAT_T   n,
                                  SUPV_T*   y,
                                  COMP_T    eta);
    virtual MySolver& train_batch(COMP_T*   data,
                                  DAT_DIM_T d,
                                  N_DAT_T*  x,
                                  N_DAT_T   n,
                                  SUPV_T*   y,
                                  COMP_T    eta);
    virtual COMP_T    compute_obj(COMP_T*   data,
                                  DAT_DIM_T d,
                                  N_DAT_T   n,
                                  SUPV_T*   y);
    virtual COMP_T    compute_obj(COMP_T*   data,
                                  DAT_DIM_T d,
                                  N_DAT_T*  x,
                                  N_DAT_T   n,
                                  SUPV_T*   y);
    

private:
    /// Parameters of the training process (internal/external)
    MyParam*  my_param;
    /// parameters of the hyperplane (internal)
    COMP_T*   w;
    /// bias of the hyperplane
    COMP_T    b;
    /// probability of each class (internal)
    COMP_T*   p;

    bool      alloc_my_param;
    // external
    N_DAT_T*  x_pos;
    N_DAT_T   n_x_pos;
    N_DAT_T*  x_neg;
    N_DAT_T   n_x_neg;

    // Buffer variables for efficient training
    COMP_T*   p_margin_pos;   // (internal)
    COMP_T*   p_margin_mid;   // (internal)
    COMP_T*&  p_margin_neg;   // (internal)
    COMP_T*   term_loss;      // (internal/external)
    std::vector<std::pair<COMP_T, SUPV_T>> supp_p_max;
    std::vector<std::pair<COMP_T, SUPV_T>> supp_p_min;

    MySolver& full_train(unsigned int& i,
                         unsigned int  n_iter,
                         DAT_DIM_T     dim,
                         N_DAT_T*      x,
                         N_DAT_T       n_sample,
                         SUPV_T        n_label,
                         COMP_T        eta0);
    MySolver& fine_train(unsigned int& i,
                         unsigned int  n_iter_fine,
                         DAT_DIM_T     dim,
                         N_DAT_T*      x,
                         N_DAT_T       n_sample,
                         SUPV_T        n_label);

    COMP_T    compute_score(COMP_T* dat_i, DAT_DIM_T d) const;
    COMP_T    compute_norm(DAT_DIM_T d) const;
    COMP_T    info_gain(N_DAT_T n_sample, SUPV_T n_label) const;

    // Inherited functions
    virtual MySolver* duplicate_this() { return new MySolver(*this); }
    virtual MySolver& assign_to_this(void* _some) {
        MySolver* some = (MySolver*)_some;
        std::memcpy(w, some->w, gd_param->dimension() * sizeof(COMP_T));
        b = some->b;
        std::memcpy(p, some->p, stat.num_of_labels() * sizeof(COMP_T));
        return *this;
    }
}; // Class MySolver


inline MySolver& MySolver::solve(N_DAT_T* _x_pos, N_DAT_T& _n_x_pos,
                                 N_DAT_T* _x_neg, N_DAT_T& _n_x_neg) {
    x_pos = _x_pos;
    x_neg = _x_neg;
    solve();
    _n_x_pos = n_x_pos;
    _n_x_neg = n_x_neg;
    return *this;
}

inline COMP_T MySolver::entropy() {
    return (COMP_T)stat.entropy();
}

inline SUPV_T MySolver::test_one(COMP_T* dat_i, DAT_DIM_T d) const {
    return compute_score(dat_i, d) > 0 ? 0 : 1;
}

inline MySolver& MySolver::reset_support_p() {
    SUPV_T n_supp_p = my_param->num_of_support_ps();
    supp_p_max.clear();
    supp_p_min.clear();
    supp_p_max.reserve(n_supp_p);
    supp_p_min.reserve(n_supp_p);
    return *this;
}

inline COMP_T MySolver::compute_learning_rate(COMP_T eta0, unsigned long t) {
    return eta0 / (1 + my_param->regul_coef() * eta0 * t);
}

# endif
