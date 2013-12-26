# ifndef _MYFOREST_HPP
# define _MYFOREST_HPP

# include "my_typedefs.h"
# include "MyTree.hpp"

class MyForest {
public:
    class MyForestParam {
    public:
        MyForestParam(N_DAT_T   _n_sub_set = 200,
                      DAT_DIM_T _n_sub_dim = 1):
                      n_sub_set(_n_sub_set),
                      n_sub_dim(_n_sub_dim) {}
        ~MyForestParam();
        MyForestParam& num_of_bootstraps(N_DAT_T _n_sub_set)
            { n_sub_set = _n_sub_set; return *this; }
        MyForestParam& num_of_subspace(DAT_DIM_T _n_sub_dim)
            { n_sub_dim = _n_sub_dim; return *this; }
        N_DAT_T        num_of_bootstraps() const { return n_sub_set; }
        DAT_DIM_T      num_of_subspaces()  const { return n_sub_dim; }
    private:
        N_DAT_T   n_sub_set;
        DAT_DIM_T n_sub_dim;
    };

    MyForest(char         _verbosity            = 1,
             DAT_DIM_T    _dimension            = 2,
             N_DAT_T      _n_sub_set            = 200,
             DAT_DIM_T    _n_sub_dim            = 1,
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
    ~MyForest();

    MyForest& training_files(char** _train_files)
        { train_files = _train_files; return *this; }
    MyForest& train();
    MyForest& test(COMP_T* data, N_DAT_T n, SUPV_T* y);
    SUPV_T    test_one(COMP_T* data, DAT_DIM_T dim, N_DAT_T* distrib = NULL);

private:
    MyForest& rand_index(N_DAT_T* x);
    MyForest& read_data(char* train_file);

    MyTree*** trees;
    char**    train_files;

    SUPV_T    n_label;

    COMP_T*   data;     ///< data buffer (external)
    N_DAT_T   n;        ///< total number of samples in the buffer
    SUPV_T*   y;        ///< labels (external)

    GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam   gd_param;
    SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam sgd_param;
    MySolver::MyParam                                 my_param;
    MyTree::MyTreeParam                               my_tree_param;
    MyForestParam                                     my_forest_param;
};


# endif
