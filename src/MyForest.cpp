# include <iostream>
# include <sstream>
# include <fstream>
# include <cstdlib>
# include <ctime>

# include "MyForest.hpp"


MyForest::MyForest(char         _verbosity,
                   DAT_DIM_T    _dimension,
                   N_DAT_T      _n_sub_set,
                   DAT_DIM_T    _n_sub_dim,
                   COMP_T       _min_entropy,
                   SUPV_T       _max_depth,
                   COMP_T       _lambda,
                   unsigned int _n_iter,
                   unsigned int _n_iter_fine,
                   COMP_T       _err,
                   bool         _show_p_each_iter,
                   COMP_T       _eta0,
                   N_DAT_T      _s_batch,
                   unsigned int _n_epoch,
                   float        _eta0_try_sample_rate,
                   COMP_T       _eta0_try_1st,
                   COMP_T       _eta0_try_factor,
                   bool         _show_obj_each_iter):
          trees(NULL),
          train_files(NULL),
          n_label(0),
          data(NULL),
          n(0),
          y(NULL),
          gd_param(_dimension,
                   _verbosity - 1,
                   _eta0,
                   200,
                   1e-8,
                   _eta0_try_sample_rate,
                   _eta0_try_1st,
                   _eta0_try_factor,
                   _show_obj_each_iter),
          sgd_param(_s_batch, _n_epoch),
          my_param(_verbosity,
                   _lambda,
                   _n_iter,
                   _n_iter_fine,
                   _err,
                   _show_p_each_iter),
          my_tree_param(_min_entropy, _max_depth),
          my_forest_param(_n_sub_set, _n_sub_dim) {
    trees = new MyTree**[_n_sub_dim];
    for (DAT_DIM_T i = 0; i < _n_sub_dim; ++i) {
        trees[i] = new MyTree*[_n_sub_set];
        MyTree** trees_subdim = trees[i];
        for (N_DAT_T j = 0; j < _n_sub_set; ++j) {
            trees_subdim[j] = new MyTree(gd_param,
                                         sgd_param,
                                         my_param,
                                         my_tree_param);
        }
    }
    std::srand(std::time(NULL));
}

MyForest::~MyForest() {
    delete[] data;
    delete[] y;
    DAT_DIM_T n_sub_dim = my_forest_param.num_of_subspaces();
    N_DAT_T   n_sub_set = my_forest_param.num_of_bootstraps();
    for (DAT_DIM_T i = 0; i < n_sub_dim; ++i) {
        MyTree** trees_subdim = trees[i];
        for (N_DAT_T j = 0; j < n_sub_set; ++j) {
            delete trees_subdim[j];
        }
        delete[] trees_subdim;
    }
    delete[] trees;
}

MyForest& MyForest::train() {
    DAT_DIM_T n_sub_dim = my_forest_param.num_of_subspaces();
    N_DAT_T   n_sub_set = my_forest_param.num_of_bootstraps();
    n_label = 0;
    for (DAT_DIM_T i = 0; i < n_sub_dim; ++i) {
        read_data(train_files[i]);
        MyTree** trees_subdim = trees[i];
        N_DAT_T* bootstrap_x  = new N_DAT_T[n];
        for (N_DAT_T j = 0; j < n_sub_set; ++j) {
            rand_index(bootstrap_x);
            trees_subdim[j]->train(data, n, y, bootstrap_x, n);
        }
        delete[] bootstrap_x;
    }
    return *this;
}

MyForest& MyForest::test(COMP_T* data, N_DAT_T n, SUPV_T* y) {
    DAT_DIM_T dim     = gd_param.dimension();
    COMP_T*   dat_i   = data;
    N_DAT_T*  distrib = new N_DAT_T[n_label];
    for (N_DAT_T i = 0; i < n; ++i, dat_i += dim) {
        y[i] = test_one(dat_i, dim, distrib);
    }
    delete[] distrib;
    return *this;
}

SUPV_T MyForest::test_one(COMP_T* data, DAT_DIM_T dim, N_DAT_T* distrib) {
    DAT_DIM_T n_sub_dim     = my_forest_param.num_of_subspaces();
    N_DAT_T   n_sub_set     = my_forest_param.num_of_bootstraps();
    bool      alloc_distrib = !distrib;
    if (alloc_distrib) distrib = new N_DAT_T[n_label];
    std::memcpy(distrib, 0, n_label * sizeof(N_DAT_T));
    for (DAT_DIM_T i = 0; i < n_sub_dim; ++i) {
        MyTree** trees_subdim = trees[i];
        for (N_DAT_T j = 0; j < n_sub_set; ++j) {
            LabelStat<SUPV_T, N_DAT_T>& stat = 
                trees_subdim[j]->test_distrib(data, dim);
            SUPV_T n_sublabel = stat.num_of_labels();
            for (SUPV_T k = 0; k < n_sublabel; ++k) {
                distrib[stat.label(k) - 1] += stat.num_of_samples_with_label(k);
            }
        }
    }
    SUPV_T  label          = 0;
    N_DAT_T max_n_of_label = 0;
    for (SUPV_T i = 0; i < n_label; ++i) {
        if (distrib[i] > max_n_of_label) {
            max_n_of_label = distrib[i];
            label = i + 1;
        }
    }
    if (alloc_distrib) delete[] distrib;
    return label;
}

MyForest& MyForest::rand_index(N_DAT_T* x) {
    for (N_DAT_T i = 0; i < n; ++i) {
        x[i] = std::rand() % n;
    }
    return *this;
}

MyForest& MyForest::read_data(char* train_file) {
    std::cout << "Scanning examples..." << std::flush;
    std::ifstream file(train_file);
    if (!file.is_open()) {
        std::cerr << "\nFailed opening file: " << train_file << std::endl;
        std::exit(1);
    }
    n = 0; DAT_DIM_T dim = 0;
    char line_str[16384];
    std::istringstream line;
    while (file.getline(line_str, 16384)) {
        int i;
        int line_str_len = std::strlen(line_str);
        for (i = 0; i < line_str_len && line_str[i] == ' '; ++i);
        if (std::isdigit(line_str[i])) n++;
        else continue;      // not a valid line that contains data
        for (; i < line_str_len && line_str[i] != '#'; ++i);
        if (i < line_str_len) line_str[i] = '\0';
        for (; i >= 0 && line_str[i] != ':'; --i);
        for (--i; i >= 0 && line_str[i] == ' '; --i);
        for (; i >= 0 && line_str[i] != ' '; --i);
        if (i < 0) {
            std::cerr << "\nError: Corrputed data file: Cannot find the maximun dimension of the data set" << std::endl;
            std::exit(1);
        }
        line.str(line_str + i);
        DAT_DIM_T d; line >> d;
        if (dim < d) dim = d;
    }
    gd_param.dimension(dim);
    std::cout << "Done.\nTotal samples: " << n << ", Feature dimension: " << dim
              << "\nReading samples..." << std::flush;
    
    delete[] data;
    delete[] y;
    data = new COMP_T[dim * n];
    y    = new SUPV_T[n];

    unsigned long N = dim * n;
    for (unsigned long i = 0; i < N; ++i) data[i] = 0;

    file.clear();
    file.seekg(0);

    N_DAT_T count = 0;
    while (count < n && file.getline(line_str, 16384)) {
        int i;
        int line_str_len = std::strlen(line_str);
        for (i = 0; i < line_str_len && line_str[i] == ' '; ++i);
        if (!std::isdigit(line_str[i])) continue;
        for (; i < line_str_len && line_str[i] != '#'; ++i);
        if (i < line_str_len) line_str[i] = '\0';
        line.str(line_str);
        SUPV_T label;
        line >> label;
        y[count] = label;
        if (n_label < label) n_label = label;
        DAT_DIM_T d; char comma; COMP_T x;
        COMP_T* dat_i = data + count * dim;
        while (line >> d >> comma >> x) {
            dat_i[d - 1] = x;
        }
        if (line.eof()) line.clear();
        else {
            std::cerr << "\nError: Corrputed data file: Wrong format at line "
                      << count << std::endl;
            std::exit(1);
        }
        ++count;
    }

    std::cout << "Done." << std::endl;
    file.close();
    return *this;
}
