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
                unsigned int _n_iter           = 100,
                COMP_T       _err              = 0.01,
                bool         _show_p_each_iter = false)
               :v(_v),
                lambda(_lambda),
                n_iter(_n_iter),
                err(_err),
                show_p_each_iter_(_show_p_each_iter) {}
        ~MyParam() {}
        
        MyParam& verbosity(char _v) { v = _v; return *this; }
        MyParam& regul_coef(COMP_T _lambda) { lambda = _lambda; return *this; }
        MyParam& num_of_iterations(unsigned int _n_iter)
            { n_iter = _n_iter; return *this; }
        MyParam& accuracy(COMP_T _err) { err = _err; return *this; }
        MyParam& show_p_each_iter(bool _show)
            { show_p_each_iter_ = _show; return *this; }

        char         verbosity()         const { return v; }
        COMP_T       regul_coef()        const { return lambda; }
        unsigned int num_of_iterations() const { return n_iter; }
        COMP_T       accuracy()          const { return err; }
        bool         show_p_each_iter()  const { return show_p_each_iter_; }

        MyParam& ostream_this(std::ostream& out) {
            out << "MySolver parameters:\n"
                << "\tVerbosity = " << v << "\n"
                << "\tRegularization coefficient = " << lambda << "\n"
                << "\tMax number of iterations = " << n_iter << "\n"
                << "\tStopping accuracy = " << err;
            return *this;
        }

    private:
        char         v;
        COMP_T       lambda;      ///< trade-off between regularization & loss
        unsigned int n_iter;
        COMP_T       err;
        bool         show_p_each_iter_;
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
    /// @param[in]  dat The matrix of the input data.
    /// @param[in]  n   The number of samples.
    /// @param[in]  y   The labels of every sample.
    /// @param[out] x0  A buffer of size "n" that stores the indexes of samples 
    ///                 which result in w'x+b>0.
    /// @param[out] x1  A buffer of size "n" that stores the indexes of samples 
    ///                 which result in w'x+b<0.
    /// @param[out] d0  A buffer which has the equal size of the label set that 
    ///                 stores number of samples of each label that result in 
    ///                 w'x+b>0.
    /// @param[out] d0  A buffer which has the equal size of the label set that 
    ///                 stores number of samples of each label that result in 
    ///                 w'x+b<0.
    MySolver&         solve(N_DAT_T* x_pos, N_DAT_T& n_x_pos,
                            N_DAT_T* x_neg, N_DAT_T& n_x_neg);

protected:
    /// Update the information of p.
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
    /// Test one sample to judge whether the sample lies in the positive side or 
    /// negative side of the hyperplane "w'x+b=0".
    /// 
    /// @return     1 if positive, 0 if negative.
    virtual SUPV_T    test_one(COMP_T* dat_i, DAT_DIM_T d) const;

private:
    /// Parameters of the training process
    MyParam* my_param;
    /// parameters of the hyperplane
    COMP_T*  w;
    /// bias of the hyperplane
    COMP_T   b;
    /// probability of each class
    COMP_T*  p;

    bool     alloc_p;

    // Buffer variables for efficient training
    SUPV_T*  margin_x;  // An array of the same size as the WHOLE data set.
                        // "margin_x[i]" is 0 if x_i has score < 1;
                        // 1 if -1<score<1; 2 if score > 1.
    COMP_T** p_margin;
    COMP_T*  term2;

    MySolver& initialize(DAT_DIM_T d,
                         SUPV_T    n_label,
                         N_DAT_T   n_sample,
                         N_DAT_T* x_pos, N_DAT_T& n_x_pos,
                         N_DAT_T* x_neg, N_DAT_T& n_x_neg);

    COMP_T compute_score(COMP_T* dat_i, DAT_DIM_T d) const;
    COMP_T compute_norm(DAT_DIM_T d) const;

    // Inherited functions
    virtual MySolver* duplicate_this() { return new MySolver(*this); }
}; // Class MySolver


inline COMP_T MySolver::compute_learning_rate(COMP_T eta0, N_DAT_T t) {
    return eta0 / (1 + my_param->regul_coef() * eta0 * t);
}

inline SUPV_T MySolver::test_one(COMP_T* dat_i, DAT_DIM_T d) const {
    return compute_score(dat_i, d) > 0;
}


# endif
