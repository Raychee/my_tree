# include <iostream>
# include <iomanip>
# include <sstream>
# include <fstream>
# include <cstring>
# include <cstdlib>
# include <cstdio>

# include "my_typedefs.h"
# include "my_lib.hpp"
# include "MyTree.hpp"

enum code {
    ERROR,
    HELP,
    CONFIG_FILE,
    LOG_FILE,
    TREE_FILE,
    GD_VERBOSITY,
    GD_INIT_LEARNING_RATE,
    GD_N_ITER,
    GD_ERR,
    GD_MIN_N_SUBSAMPLE,
    GD_LEARNING_RATE_TRY_SAMPLE_RATE,
    GD_LEARNING_RATE_TRY_1ST,
    GD_LEARNING_RATE_TRY_FACTOR,
    GD_SHOW_OBJ_EACH_ITER,
    SGD_S_BATCH,
    SGD_N_EPOCH,
    MY_VERBOSITY,
    MY_REG_COEF,
    MY_N_TRIAL,
    MY_N_TRAIN,
    MY_N_ITER,
    MY_N_ITER_FINE,
    MY_N_SUPP_P,
    MY_N_INIT_SUPP_P,
    MY_N_INC_SUPP_P,
    MY_SUPP_P_INC_INTV,
    MY_BIAS_LEARNING_RATE_FACTOR,
    MY_ERR,
    MY_REG_BIAS,
    MY_SHOW_P_EACH_ITER,
    MY_TREE_MIN_ENTROPY,
    MY_TREE_MAX_DEPTH
};

void read_data(const char* data_file,
               COMP_T*& X, DAT_DIM_T& d, N_DAT_T& N, SUPV_T*& Y);
void read_args(int   argc,      const char** argv,
               char* data_file, char*        log_file, char*      tree_dir,
               GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   gd_param,
               SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& sgd_param,
               MySolver::MyParam&                                 my_param,
               MyTree::MyTreeParam&                               my_tree_param);
void assert_arg(bool oops);
code parse_arg_short(const char* arg);
code parse_arg_long(const char* arg);
void read_config(const char*                                        config_file,
                 GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   gd_param,
                 SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& sgd_param,
                 MySolver::MyParam&                                 my_param,
                 MyTree::MyTreeParam&                               my_tree_param);
void print_help();

int main(int argc, const char** argv)
{

    char data_file[1024], log_file[1024], tree_dir[1024];
    COMP_T* X; DAT_DIM_T D; N_DAT_T N; SUPV_T* Y;

    GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam   gd_param;
    SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam sgd_param;
    MySolver::MyParam                                 my_param;
    MyTree::MyTreeParam                               my_tree_param;

    read_args(argc, argv, data_file, log_file, tree_dir,
              gd_param, sgd_param, my_param, my_tree_param);
    read_data(data_file, X, D, N, Y);
    gd_param.dimension(D);

    std::ofstream log_f;
    if (log_file[0] != '\0') {
        log_f.open(log_file);
        if (!log_f.is_open()) {
            std::cerr << "\nFailed opening file: " << log_file << std::endl;
            std::exit(1);
        }
        char v = my_tree_param.verbosity();
        if (v > 2) gd_param.ostream_of_training_process(log_f);
        else if (v > 1) my_param.ostream_of_training_process(log_f);
        else if (v > 0) my_tree_param.ostream_of_training_result(log_f);
    }
    
    MyTree tree(gd_param, sgd_param, my_param, my_tree_param);

    tree.train(X, N, Y);
    tree.save_this(tree_dir);

    log_f.close();

    delete[] X;
    delete[] Y;
    return 0;
}

void read_data(const char* data_file,
               COMP_T*& X, DAT_DIM_T& D, N_DAT_T& N, SUPV_T*& Y) {
    std::cout << "Scanning examples..." << std::flush;
    std::ifstream file(data_file);
    if (!file.is_open()) {
        std::cerr << "\nFailed opening file: " << data_file << std::endl;
        std::exit(1);
    }
    N = 0; D = 0;
    char line_str[16384];
    std::istringstream line;
    while (file.getline(line_str, 16384)) {
        int i = 0, line_str_len = std::strlen(line_str);
        for (i = 0; i < line_str_len && line_str[i] == ' '; ++i);
        if (std::isdigit(line_str[i])) ++N;
        else continue;
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
        if (D < d) D = d;
    }
    std::cout << "Done.\nTotal samples: " << N << ", Feature dimension: " << D
              << "\nReading samples..." << std::flush;
    X = new COMP_T[D * N];
    Y = new SUPV_T[N];
    file.clear();
    file.seekg(0);
    N_DAT_T n = 0;
    while (n < N && file.getline(line_str, 16384)) {
        int i = 0, line_str_len = std::strlen(line_str);
        for (i = 0; i < line_str_len && line_str[i] == ' '; ++i);
        if (!std::isdigit(line_str[i])) continue;
        for (; i < line_str_len && line_str[i] != '#'; ++i);
        if (i < line_str_len) line_str[i] = '\0';
        line.str(line_str);
        line >> Y[n];
        DAT_DIM_T d; char colon; COMP_T x;
        COMP_T* X_i = X + n * D;
        while (line >> d >> colon >> x) {
            X_i[d - 1] = x;
        }
        if (line.eof()) line.clear();
        else {
            std::cerr << "\nError: Corrputed data file: Wrong format at line "
                      << n << std::endl;
            std::exit(1);
        }
        ++n;
    }
    std::cout << "Done." << std::endl;
    file.close();
}

void read_args(int   argc,      const char** argv,
               char* data_file, char*        log_file,  char*     tree_dir,
               GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   gd_param,
               SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& sgd_param,
               MySolver::MyParam&                                 my_param,
               MyTree::MyTreeParam&                               my_tree_param) {
    int i = 0;
    if (argc < 2) { print_help(); exit(0); }
    while (++i < argc) {
        if (argv[i][0] == '-') {
            bool last_arg = i + 1 >= argc;
            code arg_code;
            if (argv[i][1] == '-') arg_code = parse_arg_long(argv[i] + 2);
            else arg_code = parse_arg_short(argv[i] + 1);
            switch(arg_code) {
                case HELP:
                    print_help(); exit(0);
                case CONFIG_FILE: assert_arg(last_arg); ++i;
                    read_config(argv[i], gd_param, sgd_param, my_param, my_tree_param);
                    break;
                case LOG_FILE: assert_arg(last_arg); ++i;
                    std::strcpy(log_file, argv[i]);
                    break;
                case TREE_FILE: assert_arg(last_arg); ++i;
                    std::strcpy(tree_dir, argv[i]);
                    break;
                default : std::cerr << "Unrecognized option " << argv[i] << "!\n";
                          exit(1);
            }
        }
        else {
            std::strcpy(data_file, argv[i]);
        }
    }
    assert_arg(!data_file[0]);
    // for (i = 1; i < argc && argv[i][0] == '-'; ++i) {
    //     switch (parse_arg(argv[i])) {
    //         case 0  : print_help(); exit(0);
    //         case 1  : ++i; gd_param.verbosity(argv[i][0] - '0');
    //         case 2  : ++i; my_param.regul_coef(std::atof(argv[i]));
    //         case 3  : ++i; my_param.regul_coef(std::atof(argv[i])); break;
    //         case 5  : ++i; sgd_param.num_of_epoches(std::atoi(argv[i]));
    //         case 6  : ++i; sgd_param.num_of_epoches(std::atoi(argv[i])); break;
    //         case 11 : ++i; gd_param.init_learning_rate(std::atof(argv[i]));
    //         case 12 : ++i; gd_param.init_learning_rate(std::atof(argv[i])); break;
    //         case 14 : ++i; gd_param.learning_rate_1st_try(std::atof(argv[i]));
    //         case 15 : ++i; gd_param.learning_rate_1st_try(std::atof(argv[i])); break;
    //         case 17 : ++i; gd_param.learning_rate_try_factor(std::atof(argv[i]));
    //         case 18 : ++i; gd_param.learning_rate_try_factor(std::atof(argv[i])); break;
    //         default:
    //             std::cerr << "Unrecognized option " << argv[i] << "! Please type \"labeltree_train -h\" for help.\n";
    //             exit(1);
    //     }
    // }
}

void assert_arg(bool oops) {
    if (oops) {
        std::cerr << "Not enough input parameters!\n";
        exit(1);
    }
}

code parse_arg_short(const char* arg) {
    if (arg[1] != '\0') return ERROR;
    switch (arg[0]) {
        case 'h': return HELP;
        case 'o': return TREE_FILE;
        // case 'v': return 1;
        // case 'n': return 5;
        // case 'f': return 8;
        // case 'e': return 11;
        // case 'b': return 20;
        // case 'o': return 256;
        default : return ERROR;
    }
}

code parse_arg_long(const char* arg) {
    if (!std::strcmp(arg, "help"))                 return HELP;
    // if (!std::strcmp(arg, "verbosity"))            return 1;
    // if (!std::strcmp(arg, "lambda"))               return 2;
    // if (!std::strcmp(arg, "svm-lambda"))           return 3;
    // if (!std::strcmp(arg, "tree-lambda"))          return 4;
    // if (!std::strcmp(arg, "epoch"))                return 5;
    // if (!std::strcmp(arg, "svm-epoch"))            return 6;
    // if (!std::strcmp(arg, "tree-epoch"))           return 7;
    // if (!std::strcmp(arg, "comp-obj"))             return 8;
    // if (!std::strcmp(arg, "svm-comp-obj"))         return 9;
    // if (!std::strcmp(arg, "tree-comp-obj"))        return 10;
    // if (!std::strcmp(arg, "eta0"))                 return 11;
    // if (!std::strcmp(arg, "svm-eta0"))             return 12;
    // if (!std::strcmp(arg, "tree-eta0"))            return 13;
    // if (!std::strcmp(arg, "eta0-1st-try"))         return 14;
    // if (!std::strcmp(arg, "svm-eta0-1st-try"))     return 15;
    // if (!std::strcmp(arg, "tree-eta0-1st-try"))    return 16;
    // if (!std::strcmp(arg, "eta0-try-factor"))      return 17;
    // if (!std::strcmp(arg, "svm-eta0-try-factor"))  return 18;
    // if (!std::strcmp(arg, "tree-eta0-try-factor")) return 19;
    // if (!std::strcmp(arg, "nary"))                 return 20;
    // if (!std::strcmp(arg, "out"))                  return 256;
    // if (!std::strcmp(arg, "out-svm"))              return 257;
    if (!std::strcmp(arg, "config"))                 return CONFIG_FILE;
    if (!std::strcmp(arg, "log"))                    return LOG_FILE;
    return ERROR;
}

void read_config(const char*                                        config_file,
                 GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   gd_param,
                 SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& sgd_param,
                 MySolver::MyParam&                                 my_param,
                 MyTree::MyTreeParam&                               my_tree_param) {
    std::cout << "Reading configs..." << std::flush;
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "\nFailed opening file: " << config_file << std::endl;
        std::exit(1);
    }
    char line_str[1024];
    while (file.getline(line_str, 1024)) {
        char* str_param, * str_value;
        for (str_value = line_str; *str_value != '\0' && *str_value != '=' ; ++str_value);
        if (*str_value == '\0') continue;
        *(str_value++) = '\0';
        str_param = strtostr(line_str);
        if (!std::strcmp(str_param, "GD_verbosity"))
            { gd_param.verbosity(strto<int>(str_value)); }
        else if (!std::strcmp(str_param, "GD_initial_learning_rate"))
            { gd_param.init_learning_rate(strto<COMP_T>(str_value)); }
        else if (!std::strcmp(str_param, "GD_num_of_iterations"))
            { gd_param.num_of_iterations(strto<unsigned int>(str_value)); }
        else if (!std::strcmp(str_param, "GD_accuracy"))
            { gd_param.accuracy(strto<COMP_T>(str_value)); }
        else if (!std::strcmp(str_param, "GD_min_num_of_subsamples"))
            { gd_param.min_num_of_subsamples(strto<N_DAT_T>(str_value)); }
        else if (!std::strcmp(str_param, "GD_initial_learning_rate_try_subsample_rate"))
            { gd_param.learning_rate_sample_rate(strto<float>(str_value)); }
        else if (!std::strcmp(str_param, "GD_initial_learning_rate_1st_try"))
            { gd_param.learning_rate_1st_try(strto<COMP_T>(str_value)); }
        else if (!std::strcmp(str_param, "GD_initial_learning_rate_try_factor"))
            { gd_param.learning_rate_try_factor(strto<COMP_T>(str_value)); }
        else if (!std::strcmp(str_param, "GD_show_objective_each_iteration"))
            { gd_param.show_obj_each_iteration(strto<bool>(str_value)); }
        else if (!std::strcmp(str_param, "SGD_size_of_batches"))
            { sgd_param.size_of_batches(strto<N_DAT_T>(str_value)); }
        else if (!std::strcmp(str_param, "SGD_num_of_epoches"))
            { sgd_param.num_of_epoches(strto<unsigned int>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_verbosity"))
            { my_param.verbosity(strto<int>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_regularization_strength"))
            { my_param.regul_coef(strto<COMP_T>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_num_of_trials"))
            { my_param.num_of_trials(strto<unsigned int>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_num_of_trainings"))
            { my_param.num_of_trainings(strto<unsigned int>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_num_of_iterations"))
            { my_param.num_of_iterations(strto<unsigned int>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_num_of_fine_tuning_iterations"))
            { my_param.num_of_fine_tuning(strto<unsigned int>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_supporting_p_ratio"))
            { my_param.support_p_ratio(strto<float>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_initial_supporting_p_ratio"))
            { my_param.initial_support_p_ratio(strto<float>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_interval_of_adding_support_ps"))
            { my_param.support_p_incre_interval(strto<float>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_num_of_adding_supporting_ps"))
            { my_param.num_of_incre_support_ps(strto<SUPV_T>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_bias_learning_rate_factor"))
            { my_param.bias_learning_rate_factor(strto<COMP_T>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_accuracy"))
            { my_param.accuracy(strto<COMP_T>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_regularized_bias"))
            { my_param.regul_bias(strto<bool>(str_value)); }
        else if (!std::strcmp(str_param, "SOLVER_show_ps_each_iteration"))
            { my_param.show_p_each_iter(strto<bool>(str_value)); }
        else if (!std::strcmp(str_param, "TREE_verbosity"))
            { my_tree_param.verbosity(strto<int>(str_value)); }
        else if (!std::strcmp(str_param, "TREE_min_entropy"))
            { my_tree_param.min_entropy(strto<COMP_T>(str_value)); }
        else if (!std::strcmp(str_param, "TREE_max_depth"))
            { my_tree_param.max_depth(strto<SUPV_T>(str_value)); }
    }
    file.close();
    std::cout << "Done." << std::endl;
}



void print_help() {
    std::cout << "My Tree: \n";
    std::cout << "Usage:\n    mytree_train [options] data_file\n";
    std::cout << "Arguments:\n";
    std::cout << "    data_file\n        file with training data.\n";
    std::cout << "Options:\n";
    std::cout << "    --help, -h\n";
    std::cout << "        Show this help and exit.\n";
    std::cout << "    --verbosity, -v 1\n";
    std::cout << "        Verbosity(0-4).\n";
    std::cout << "    --lambda 0.01\n";
    std::cout << "        Regular term coefficent in one-vs-all svm classifiers and\n";
    std::cout << "        label tree.\n";
    std::cout << "    --svm-lambda 0.01\n";
    std::cout << "        Regular term coefficent in one-vs-all svm classifiers.\n";
    std::cout << "    --tree-lambda 0.01\n";
    std::cout << "        Regular term coefficent in the label tree.\n";
    std::cout << "    --epoch, -n 5\n";
    std::cout << "        Number of epoches during training (set for both one-vs-alls and label\n";
    std::cout << "        tree).\n";
    std::cout << "    --svm-epoch 5\n";
    std::cout << "        Number of epoches during training of one-vs-alls.\n";
    std::cout << "    --tree-epoch 5\n";
    std::cout << "        Number of epoches during training of label tree.\n";
    std::cout << "    --comp-obj, -f\n";
    std::cout << "        Compute and echo the value of object function after each epoch (the\n";
    std::cout << "        same as --svm-comp-obj --tree-comp-obj).\n";
    std::cout << "    --svm-comp-obj\n";
    std::cout << "        Compute and echo the value of object function after each epoch during\n";
    std::cout << "        one-vs-alls' training.\n";
    std::cout << "    --tree-comp-obj\n";
    std::cout << "        Compute and echo the value of object function after each epoch\n";
    std::cout << "        during label tree's training.\n";
    std::cout << "    --eta0, -e 0\n";
    std::cout << "        The initial learning rate of the one-vs-alls' and label tree's training \n";
    std::cout << "        algorithm. (the same as --svm-eta0 0 --tree-eta0 0)\n";
    std::cout << "        If 0, then the algorithm will find one automatically according to the \n";
    std::cout << "        setting of its first guess and the factor to multiply after each guess, \n";
    std::cout << "        which can be specified using -eta0-1st-try and -eta0-try-factor \n";
    std::cout << "        respectively.\n";
    std::cout << "    --svm-eta0 0\n";
    std::cout << "        The initial learning rate of the one-vs-alls' training algorithm.\n";
    std::cout << "    --tree-eta0 0\n";
    std::cout << "        The initial learning rate of the label tree's training algorithm.\n";
    std::cout << "    --eta0-1st-try 0.1\n";
    std::cout << "        The first guess of the initial learning rate of the one-vs-alls' and \n";
    std::cout << "        label tree's training algorithm. (Only take effect when --eta0 0)\n";
    std::cout << "    --svm-eta0-1st-try 0.1\n";
    std::cout << "        The first guess of the initial learning rate of the one-vs-alls'\n";
    std::cout << "        training algorithm. (Only take effect when --svm-eta0 0)\n";
    std::cout << "    --tree-eta0-1st-try 0.1\n";
    std::cout << "        The first guess of the initial learning rate of the label tree's\n";
    std::cout << "        training algorithm. (Only take effect when --tree-eta0 0)\n";
    std::cout << "    --eta0-try-factor 3\n";
    std::cout << "        The factor that the learning rate multiplies after each guess of the \n";
    std::cout << "        initial learning rate of the one-vs-alls' and label tree's training \n";
    std::cout << "        algorithm. (Only take effect when --eta0 0)\n";
    std::cout << "    --svm-eta0-try-factor 3\n";
    std::cout << "        The factor that the learning rate multiplies after each guess of the\n";
    std::cout << "        initial learning rate of the one-vs-alls' training algorithm. (Only take\n";
    std::cout << "        effect when --eta0 0)\n";
    std::cout << "    --tree-eta0-try-factor 3\n";
    std::cout << "        The factor that the learning rate multiplies after each guess of the\n";
    std::cout << "        initial learning rate of the label tree's training algorithm. (Only take\n";
    std::cout << "        effect when --eta0 0)\n";
    std::cout << "    --nary, -b 2\n";
    std::cout << "        Number of branches of each tree node in the label tree.\n";
}
