# ifndef _MYTREE_HPP
# define _MYTREE_HPP

# include <vector>
# include <map>

# include "my_typedefs.h"
# include "SGD.h"
# include "Tree.hpp"
# include "Histogram.hpp"

class MySolver : public SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T> {
public:
    class MyParam {
    public:
        MyParam(SUPV_T _s_labelset = 0,
                COMP_T _lambda     = 0.5)
               :lambda(_lambda),
                s_labelset(_s_labelset) {}
        ~MyParam() {}
        MyParam& num_of_labels(SUPV_T _s_labelset) {
            if (s_labelset) {
                std::cerr << "WARNING: MyTree: "
                          << "Size of the label set has already been specified."
                          << " Nothing is changed." << std::endl;
            }
            else s_labelset = _s_labelset;
            return *this;
        }
        MyParam& regul_coef(COMP_T _lambda) { lambda = _lambda; return *this; }
        SUPV_T   num_of_labels() const      { return s_labelset; }
        COMP_T   regul_coef()    const      { return lambda; }
    private:
        COMP_T lambda;      ///< trade-off between regularization & loss
        SUPV_T s_labelset;  ///< size of the label set
    };//class MyParam

    MySolver(Param&   _param,
             MyParam& _my_param,
             N_DAT_T* _x,
             N_DAT_T* _s_x,
             N_DAT_T* _d);
    MySolver(MySolver& some);
    virtual ~MySolver();

    virtual COMP_T    compute_obj(COMP_T* dat, N_DAT_T n, SUPV_T* y);
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
    MySolver&         solve(COMP_T* dat, N_DAT_T n, SUPV_T* y,
                            N_DAT_T* x0 = NULL, N_DAT_T* x1 = NULL,
                            N_DAT_T* d0 = NULL, N_DAT_T* d1 = NULL);

protected:
    virtual MySolver& train_one(COMP_T* dat, N_DAT_T i,
                                N_DAT_T n, SUPV_T* y);
    virtual SUPV_T    test_one(COMP_T* dat_i) const;
    virtual COMP_T    compute_learning_rate();
    /// Update the information of p
    MySolver&         update_p(N_DAT_T* d0);

private:
    /// Parameters of the training process
    MyParam* my_param;
    /// parameters of the hyperplane
    COMP_T*  w;
    /// bias of the hyperplane
    COMP_T   b;
    /// indexes of samples within the node
    N_DAT_T* x;
    /// number of samples within the node
    N_DAT_T  s_x;

    /// Distributions of each class in this node
    Histogram<SUPV_T, N_DAT_T> d;

    /// Conditional probabilistic distributions of each class in this node
    Histogram<SUPV_T, COMP_T>  p;

    COMP_T compute_score(COMP_T* dat_i) const;
    COMP_T compute_loss(COMP_T* dat_i, unsigned int y) const;
    COMP_T compute_norm() const;

    virtual MySolver* get_temp_dup() { return new MySolver(*this); }
}; // Class MySolver

class MyTree : public Tree<MySolver*> {
public:
    MyTree(DAT_DIM_T    _dimension,
           SUPV_T       _s_labelset,
           COMP_T       _lambda              = 0.5,
           char         _verbosity           = 1,
           COMP_T       _eta0                = 0,
           unsigned int _n_epoch             = 5,
           COMP_T       _eta0_1st_try        = 0.1,
           COMP_T       _eta0_try_factor     = 3,
           bool         _comp_obj_each_epoch = false);
    MyTree& train(COMP_T* dat, N_DAT_T n, SUPV_T* y);

private:
    SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::Param sgd_param;
    MySolver::MyParam                              my_param;
};


# endif