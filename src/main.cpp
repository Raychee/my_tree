# include <iostream>
# include <iomanip>
# include <sstream>
# include <fstream>
# include <cstring>
# include <cstdlib>
# include <cstdio>

# include "my_typedefs.h"
# include "MyTree.hpp"


void read_data(const char* data_file,
               COMP_T*& X, DAT_DIM_T& d, N_DAT_T& N, SUPV_T*& Y);
void read_args(int argc, const char** argv,
               GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam&   gd_param,
               SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& sgd_param,
               MySolver::MyParam& my_param,
               char* data_file);
int  parse_arg(const char* arg);
void print_help();


int main(int argc, const char** argv)
{

    char data_file[256];
    COMP_T* X; DAT_DIM_T D; N_DAT_T N; SUPV_T* Y, * _Y_;

    GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam   gd_param;
    SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam sgd_param;
    MySolver::MyParam                                 my_param;
    MyTree::MyTreeParam                               my_tree_param;

    // std::ofstream file("data/log.txt");
    // gd_param.ostream_of_training_process(file);
    // my_param.ostream_of_training_process(file);
    // my_tree_param.ostream_of_training_result(file);

    MyTree tree(gd_param, sgd_param, my_param, my_tree_param);

    read_args(argc, argv, gd_param, sgd_param, my_param, data_file);
    read_data(data_file, X, D, N, Y);

    gd_param.dimension(D);

    // std::cout << "N = " << N << std::endl;
    // std::cout << "D = " << D << std::endl;
    // std::cout << "[ Y X' ] = \n";
    // for (N_DAT_T i = 0; i < N; ++i) {
    //     std::cout << std::left << std::setw(6) << Y[i] << "|";
    //     for (DAT_DIM_T j = 0; j < D; ++j) {
    //         std::cout << std::right << std::setw(12) << X[i * D + j];
    //     }
    //     std::cout << std::endl;
    // }

    // N_DAT_T* x_pos = new N_DAT_T[N];
    // N_DAT_T* x_neg = new N_DAT_T[N];
    // N_DAT_T  n_x_pos, n_x_neg;

    tree.train(X, N, Y);
    

    std::cout << tree << std::endl;

    _Y_ = new SUPV_T[N];

    tree.test(X, N, _Y_);

    std::cout << "[ Y _Y_ X' ] = \n";
    N_DAT_T n_correct = 0;
    for (N_DAT_T i = 0; i < N; ++i) {
        // std::cout << std::left << std::setw(6) << Y[i] << "|" << std::setw(6) << _Y_[i] << "|";
        // for (DAT_DIM_T j = 0; j < D; ++j) {
        //     std::cout << std::right << std::setw(12) << X[i * D + j];
        // }
        // std::cout << std::endl;
        if (Y[i] == _Y_[i]) n_correct++;
    }
    std::cout << "Accuracy = " << (COMP_T)n_correct / N 
              << " (" << n_correct << "/" << N << ")" << std::endl;

    // std::cout << "Entropy = " << tree.entropy() << std::endl;
    // LabelStat<SUPV_T, N_DAT_T> stat;
    // stat.stat(Y, n_x_pos, x_pos);
    // std::cout << "Entropy of positives = " << stat.entropy() << std::endl;
    // stat.stat(Y, n_x_neg, x_neg);
    // std::cout << "Entropy of negatives = " << stat.entropy() << std::endl;

    // N_DAT_T x[30];
    // for (N_DAT_T i = 0; i < 30; ++i) {
    //     x[i] = i;
    // }
    // stat.stat(Y, 30, x);
    // std::cout << "Entropy of first 30 = " << stat.entropy() << std::endl;

    // file.close();
    delete[] X;
    delete[] Y;
    delete[] _Y_;
    // delete[] x_pos;
    // delete[] x_neg;
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
        if (std::isdigit(line_str[0])) N++;
        else continue;      // not a valid line that contains data
        int i, line_str_len = std::strlen(line_str);
        for (i = 0; i < line_str_len; ++i) {
            if (line_str[i] == '#') {
                line_str[i] = '\0';
                break;
            }
        }
        for (; i >= 0 && line_str[i] != ':'; --i);
        for (; i >= 0 && line_str[i] == ' '; --i);
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
    std::memset(X, 0, sizeof(COMP_T) * D * N);

    file.clear();
    file.seekg(0);

    N_DAT_T n = 0;
    while (n < N && file.getline(line_str, 16384)) {
        if (!std::isdigit(line_str[0])) continue;
        int line_str_len = std::strlen(line_str);
        for (int i = 0; i < line_str_len; ++i) {
            if (line_str[i] == '#') {
                line_str[i] = '\0';
                break;
            }
        }
        line.str(line_str);
        line >> Y[n];
        DAT_DIM_T d; char comma; COMP_T x;
        COMP_T* X_i = X + n * D;
        while (line >> d >> comma >> x) {
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

void read_args(int argc, const char** argv,
               GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam& gd_param,
               SGD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::SGDParam& sgd_param,
               MySolver::MyParam& my_param,
               char* data_file) {
    int i;
    for (i = 1; i < argc && argv[i][0] == '-'; ++i) {
        switch (parse_arg(argv[i])) {
            case 0  : print_help(); exit(0);
            case 1  : ++i; gd_param.verbosity(argv[i][0] - '0');
            case 2  : ++i; my_param.regul_coef(std::atof(argv[i]));
            case 3  : ++i; my_param.regul_coef(std::atof(argv[i])); break;
            case 5  : ++i; sgd_param.num_of_epoches(std::atoi(argv[i]));
            case 6  : ++i; sgd_param.num_of_epoches(std::atoi(argv[i])); break;
            case 11 : ++i; gd_param.init_learning_rate(std::atof(argv[i]));
            case 12 : ++i; gd_param.init_learning_rate(std::atof(argv[i])); break;
            case 14 : ++i; gd_param.learning_rate_1st_try(std::atof(argv[i]));
            case 15 : ++i; gd_param.learning_rate_1st_try(std::atof(argv[i])); break;
            case 17 : ++i; gd_param.learning_rate_try_factor(std::atof(argv[i]));
            case 18 : ++i; gd_param.learning_rate_try_factor(std::atof(argv[i])); break;
            default:
                std::cerr << "Unrecognized option " << argv[i] << "! Please type \"labeltree_train -h\" for help.\n";
                exit(1);
        }
    }
    if (i >= argc) {
        std::cerr << "Not enough input parameters! Please type \"main -h\" for help.\n";
        exit(1);
    }
    std::strcpy(data_file, argv[i]);
}

int parse_arg(const char* arg) {
    int arg_len = std::strlen(arg);
    if (arg_len < 2) return -1;
    if (arg_len == 2) {
        switch (arg[1]) {
            case 'h': return 0;
            case 'v': return 1;
            case 'n': return 5;
            case 'f': return 8;
            case 'e': return 11;
            case 'b': return 20;
            case 'o': return 256;
            default : return -1;
        }
    }
    if (!std::strcmp(arg, "--help"))                 return 0;
    if (!std::strcmp(arg, "--verbosity"))            return 1;
    if (!std::strcmp(arg, "--lambda"))               return 2;
    if (!std::strcmp(arg, "--svm-lambda"))           return 3;
    if (!std::strcmp(arg, "--tree-lambda"))          return 4;
    if (!std::strcmp(arg, "--epoch"))                return 5;
    if (!std::strcmp(arg, "--svm-epoch"))            return 6;
    if (!std::strcmp(arg, "--tree-epoch"))           return 7;
    if (!std::strcmp(arg, "--comp-obj"))             return 8;
    if (!std::strcmp(arg, "--svm-comp-obj"))         return 9;
    if (!std::strcmp(arg, "--tree-comp-obj"))        return 10;
    if (!std::strcmp(arg, "--eta0"))                 return 11;
    if (!std::strcmp(arg, "--svm-eta0"))             return 12;
    if (!std::strcmp(arg, "--tree-eta0"))            return 13;
    if (!std::strcmp(arg, "--eta0-1st-try"))         return 14;
    if (!std::strcmp(arg, "--svm-eta0-1st-try"))     return 15;
    if (!std::strcmp(arg, "--tree-eta0-1st-try"))    return 16;
    if (!std::strcmp(arg, "--eta0-try-factor"))      return 17;
    if (!std::strcmp(arg, "--svm-eta0-try-factor"))  return 18;
    if (!std::strcmp(arg, "--tree-eta0-try-factor")) return 19;
    if (!std::strcmp(arg, "--nary"))                 return 20;
    if (!std::strcmp(arg, "--out"))                  return 256;
    if (!std::strcmp(arg, "--out-svm"))              return 257;

    return -1;
}

void print_help() {
    std::cout << "Label Tree: \n";
    std::cout << "Usage:\n    labeltree_train [options] data_file\n";
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
