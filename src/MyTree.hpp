# ifndef _MYTREE_HPP
# define _MYTREE_HPP

# include "my_typedefs.h"
# include "MySolver.hpp"
# include "Tree.hpp"


class MyTree : public Tree<MySolver> {
public:
    MyTree(char         _verbosity            = 1,
           DAT_DIM_T    _dimension            = 2,
           COMP_T       _min_entropy          = 0.1,
           SUPV_T       _max_depth            = 0,
           COMP_T       _lambda               = 1e-4,
           unsigned int _n_iter               = 10,
           unsigned int _n_iter_fine          = 50,
           COMP_T       _err                  = 0.01,
           bool         _show_p_each_iter     = false,
           COMP_T       _eta0                 = 0,
           N_DAT_T      _s_batch              = 1,
           unsigned int _n_epoch              = 50,
           float        _eta0_try_sample_rate = 0.3,
           COMP_T       _eta0_try_1st         = 0.1,
           COMP_T       _eta0_try_factor      = 3,
           bool         _show_obj_each_iter   = false);
    MyTree& train(COMP_T* data, N_DAT_T n, SUPV_T* y,
                     N_DAT_T* x = NULL, N_DAT_T s_x = 0);
    MyTree& test(COMP_T* data, N_DAT_T n, SUPV_T* y);
    SUPV_T  test_one(COMP_T* data);
    LabelStat<SUPV_T, N_DAT_T>& 
            test_distrib(COMP_T* data);

    MyTree& min_entropy(COMP_T _min_entropy)
        { min_entropy_ = _min_entropy; return *this; }
    MyTree& max_depth(SUPV_T _max_depth) 
        { max_depth_ = _max_depth; return *this; }

    COMP_T  min_entropy() const { return min_entropy_; }
    SUPV_T  max_depth()   const { return max_depth_; }

private:
    GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam   gd_param;
    SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam sgd_param;
    MySolver::MyParam my_param;
    COMP_T            min_entropy_;
    SUPV_T            max_depth_;
};


# endif