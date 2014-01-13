# ifndef _MYTREE_HPP
# define _MYTREE_HPP

# include "my_typedefs.h"
# include "MySolver.hpp"
# include "Tree.hpp"


class MyTree : public Tree<MySolver> {
public:
    class MyTreeParam {
    public:
        MyTreeParam(char    _v            = 1,
                    COMP_T  _min_entropy  = 0.1,
                    SUPV_T  _max_depth    = 0, 
                    N_DAT_T _min_n_sample = 1)
                   :v(_v),
                    min_entropy_(_min_entropy),
                    max_depth_(_max_depth),
                    min_n_sample(_min_n_sample),
                    out_training_result(NULL) {}
        ~MyTreeParam() {}

        MyTreeParam& verbosity(char _v)
            { v = _v; return *this; }
        MyTreeParam& min_entropy(COMP_T _min_entropy)
            { min_entropy_ = _min_entropy; return *this; }
        MyTreeParam& max_depth(SUPV_T _max_depth)
            { max_depth_ = _max_depth; return *this; }
        MyTreeParam& min_num_of_samples(N_DAT_T _min_n_sample)
            { min_n_sample = _min_n_sample; return *this; }
        MyTreeParam& ostream_of_training_result(std::ostream& _out)
            { out_training_result = &_out; return *this; }

        char    verbosity()          const { return v; }
        COMP_T  min_entropy()        const { return min_entropy_; }
        SUPV_T  max_depth()          const { return max_depth_; }
        N_DAT_T min_num_of_samples() const { return min_n_sample; }
        std::ostream* ostream_of_training_result() const
            { return out_training_result; }

    private:
        char          v;
        COMP_T        min_entropy_;
        SUPV_T        max_depth_;
        N_DAT_T       min_n_sample;
        std::ostream* out_training_result;
    };

    MyTree();
    MyTree(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
           SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
           MySolver::MyParam&                                 _my_param,
           MyTreeParam&                                       _my_tree_param);
    ~MyTree() {};

    MyTree& set_param(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param);
    MyTree& set_param(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
                      SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
                      MySolver::MyParam&                                 _my_param,
                      MyTreeParam&                                       _my_tree_param);
    MyTree& train(COMP_T* data, N_DAT_T n, SUPV_T* y,
                  N_DAT_T* x = NULL, N_DAT_T s_x = 0);
    MyTree& test(COMP_T* data, N_DAT_T n, SUPV_T* y);
    SUPV_T  test_one(COMP_T* data, DAT_DIM_T dim);
    LabelStat<SUPV_T, N_DAT_T>& 
            test_distrib(COMP_T* data, DAT_DIM_T dim);

private:
    GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam*   gd_param;
    SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam* sgd_param;
    MySolver::MyParam*                                 my_param;
    MyTreeParam*                                       my_tree_param;
};


inline MyTree::MyTree(): gd_param(NULL),
                         sgd_param(NULL),
                         my_param(NULL),
                         my_tree_param(NULL) {
}

inline MyTree::MyTree(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
                      SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
                      MySolver::MyParam&                                 _my_param,
                      MyTreeParam&                                       _my_tree_param):
        gd_param(&_gd_param),
        sgd_param(&_sgd_param),
        my_param(&_my_param),
        my_tree_param(&_my_tree_param) {
}

inline MyTree& MyTree::set_param(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam& _gd_param) {
    gd_param = &_gd_param;
    return *this;
}

inline MyTree& MyTree::set_param(GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   _gd_param,
                                 SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& _sgd_param,
                                 MySolver::MyParam&                                 _my_param,
                                 MyTreeParam&                                       _my_tree_param) {
    gd_param      = &_gd_param;
    sgd_param     = &_sgd_param;
    my_param      = &_my_param;
    my_tree_param = &_my_tree_param;
    return *this;
}


# endif