# ifndef _MYSOLVER_HPP
# define _MYSOLVER_HPP

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
                unsigned int _n_iter           = 50,
                unsigned int _n_iter_fine      = 50,
                COMP_T       _err              = 0.01,
                bool         _show_p_each_iter = false)
               :v(_v),
                lambda(_lambda),
                n_iter(_n_iter),
                n_iter_fine(_n_iter_fine),
                err(_err),
                show_p_each_iter_(_show_p_each_iter),
                out_training_proc(NULL) {}
        ~MyParam() {}
        
        MyParam& verbosity(char _v) { v = _v; return *this; }
        MyParam& regul_coef(COMP_T _lambda) { lambda = _lambda; return *this; }
        MyParam& num_of_iterations(unsigned int _n_iter)
            { n_iter = _n_iter; return *this; }
        MyParam& num_of_fine_tuning(unsigned int _n_iter_fine)
            { n_iter_fine = _n_iter_fine; return *this; }
        MyParam& accuracy(COMP_T _err) { err = _err; return *this; }
        MyParam& show_p_each_iter(bool _show)
            { show_p_each_iter_ = _show; return *this; }
        MyParam& ostream_of_training_process(std::ostream& _out_training_proc)
            { out_training_proc = &_out_training_proc; return *this; }

        char          verbosity()                   const { return v; }
        COMP_T        regul_coef()                  const { return lambda; }
        unsigned int  num_of_iterations()           const { return n_iter; }
        unsigned int  num_of_fine_tuning()          const { return n_iter_fine; }
        COMP_T        accuracy()                    const { return err; }
        bool          show_p_each_iter()            const { return show_p_each_iter_; }
        std::ostream* ostream_of_training_process() const { return out_training_proc; }

        MyParam& ostream_this(std::ostream& out) {
            out << "MySolver parameters:\n"
                << "\tVerbosity = " << (int)v << "\n"
                << "\tRegularization coefficient = " << lambda << "\n"
                << "\tMax number of iterations = " << n_iter << "\n"
                << "\tStopping accuracy = " << err;
            return *this;
        }

    private:
        char          v;
        COMP_T        lambda;      ///< trade-off between regularization & loss
        unsigned int  n_iter;      ///< number of normal iterations
        unsigned int  n_iter_fine; ///< number of extra iterations for fine tuning
        COMP_T        err;
        bool          show_p_each_iter_;
        std::ostream* out_training_proc;
    };//class MyParam

    MySolver(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&  _gd_param,
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
    MySolver&         solve(N_DAT_T* x_pos, N_DAT_T& n_x_pos,
                            N_DAT_T* x_neg, N_DAT_T& n_x_neg);
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
    /// @param[out] x_pos   An array of indexes of samples on the positive side 
    ///                     of the hyperplane.
    /// @param[out] n_x_pos The number of samples in "x_pos".
    /// @param[out] x_neg   An array of indexes of samples on the negative side 
    ///                     of the hyperplane.
    /// @param[out] n_x_neg The number of samples in "x_neg".
    /// @warning    "x_pos" & "x_neg" should point to pre-allocated memory spaces 
    ///             of at least the equal size of "stat.num_of_samples()".
    /// @return     The amount of difference of "p".
    COMP_T            update_p(DAT_DIM_T d,
                               N_DAT_T*  x_pos, N_DAT_T& n_x_pos,
                               N_DAT_T*  x_neg, N_DAT_T& n_x_neg);

    // Inherited functions
    virtual COMP_T    compute_learning_rate(COMP_T eta0, N_DAT_T t);
    virtual MySolver& train_batch(COMP_T*   data,
                                  DAT_DIM_T d,
                                  N_DAT_T   n,
                                  SUPV_T*   y);
    virtual MySolver& train_batch(COMP_T*   data,
                                  DAT_DIM_T d,
                                  N_DAT_T*  x,
                                  N_DAT_T   n,
                                  SUPV_T*   y);
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
    /// Parameters of the training process (external)
    MyParam* my_param;
    /// parameters of the hyperplane (internal)
    COMP_T*  w;
    /// bias of the hyperplane
    COMP_T   b;
    /// probability of each class (internal/external)
    COMP_T*  p;

    bool     alloc_p;       ///< mark whether "p" is internal

    // Buffer variables for efficient training (internal)
    COMP_T*  p_margin_pos;
    COMP_T*  p_margin_neg;
    COMP_T*  p_margin_mid;
    COMP_T*  term2;

    MySolver& initialize(DAT_DIM_T d,
                         SUPV_T    n_label,
                         N_DAT_T   n_sample,
                         N_DAT_T* x_pos, N_DAT_T& n_x_pos,
                         N_DAT_T* x_neg, N_DAT_T& n_x_neg);

    COMP_T    compute_score(COMP_T* dat_i, DAT_DIM_T d) const;
    COMP_T    compute_norm(DAT_DIM_T d) const;

    // Inherited functions
    virtual MySolver* duplicate_this() { return new MySolver(*this); }
}; // Class MySolver

inline COMP_T MySolver::entropy() {
    return (COMP_T)stat.entropy();
}

inline COMP_T MySolver::compute_learning_rate(COMP_T eta0, N_DAT_T t) {
    return eta0 / (1 + my_param->regul_coef() * eta0 * t);
}

inline SUPV_T MySolver::test_one(COMP_T* dat_i, DAT_DIM_T d) const {
    return compute_score(dat_i, d) > 0 ? 0 : 1;
}


# endif
