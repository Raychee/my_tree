# ifndef _MYTREE_HPP
# define _MYTREE_HPP

# include <vector>
# include <map>

# include "my_typedefs.h"
# include "SGD.h"
# include "Tree.hpp"
# include "Histogram.hpp"


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