# include <iostream>
# include <iomanip>
# include <sstream>
# include <fstream>
# include <cstring>
# include <cstdlib>
# include <cstdio>

# include "my_typedefs.h"
# include "MyTree.hpp"

enum code {
    ERROR,
    HELP,
    DETAIL
};

void read_data(const char* data_file,
               COMP_T*& X, DAT_DIM_T& D, N_DAT_T& N);
void read_args(int   argc,      const char** argv,
               char* data_file, char*        tree_dir,
               bool& detail);
void assert_arg(bool oops);
code parse_arg_short(const char* arg);
code parse_arg_long(const char* arg);
void print_help();

int main(int argc, const char** argv)
{
    char data_file[1024], tree_dir[1024];
    COMP_T* X; DAT_DIM_T D; N_DAT_T N;
    bool detail = false;

    GD<COMP_T, SUPV_T, DAT_DIM_T, N_DAT_T>::GDParam   gd_param;
    MyTree tree;
    tree.set_param(gd_param);
    
    read_args(argc, argv, data_file, tree_dir, detail);
    read_data(data_file, X, D, N);
    gd_param.dimension(D);

    tree.read_this(tree_dir);

    COMP_T* x = X;
    for (N_DAT_T i = 0; i < N; ++i, x += D) {
        LabelStat<SUPV_T, N_DAT_T>& stat = tree.test_distrib(x, D);
        SUPV_T  n_label = stat.num_of_labels();
        SUPV_T  max_n_of_label;
        N_DAT_T max_n_x_of_label = 0;
        for (SUPV_T i = 0; i < n_label; ++i) {
            N_DAT_T n_x_of_label = stat.num_of_samples_with_label(i);
            if (n_x_of_label > max_n_x_of_label) {
                max_n_x_of_label = n_x_of_label;
                max_n_of_label = i;
            }
        }
        std::cout << stat.label(max_n_of_label);
        if (detail) {
            for (SUPV_T i = 0; i < n_label; ++i) {
                std::cout << " " << stat.label(i) << ":"
                          << stat.num_of_samples_with_label(i);
            }
        }
        std::cout << std::endl;
    }

    delete[] X;
    return 0;
}

void read_data(const char* data_file,
               COMP_T*& X, DAT_DIM_T& D, N_DAT_T& N) {
    std::ifstream file(data_file);
    if (!file.is_open()) {
        std::cerr << "\nFailed opening file: " << data_file << std::endl;
        std::exit(1);
    }
    N = 0;
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
    X = new COMP_T[D * N];
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
        DAT_DIM_T d; char colon; COMP_T x; SUPV_T y;
        line >> y;
        COMP_T* x_i = X + n * D;
        while (line >> d >> colon >> x) {
            x_i[d - 1] = x;
        }
        if (line.eof()) line.clear();
        else {
            std::cerr << "\nError: Corrputed data file: Wrong format at line "
                      << n << std::endl;
            std::exit(1);
        }
        ++n;
    }
    file.close();
}

void read_args(int   argc,      const char** argv,
               char* data_file, char*        tree_dir, 
               bool& detail) {
    int i = 0;
    data_file[0] = tree_dir[0] = '\0';
    if (argc < 2) { print_help(); exit(0); }
    while (++i < argc) {
        if (argv[i][0] == '-') {
            bool last_arg = i + 1 >= argc;
            int arg_code;
            if (argv[i][1] == '-') arg_code = parse_arg_long(argv[i] + 2);
            else arg_code = parse_arg_short(argv[i] + 1);
            switch(arg_code) {
                case HELP:
                    print_help(); exit(0);
                case DETAIL:
                    detail = true; break;
                default : std::cerr << "Unrecognized option " << argv[i] << "!\n";
                          exit(1);
            }
        }
        else {
            if (!data_file[0]) std::strcpy(data_file, argv[i]);
            else if (!tree_dir[0]) std::strcpy(tree_dir, argv[i]);
            else {
                std::cerr << "Too many input parameters!\n";
                exit(1);
            }
        }
    }
    assert_arg(!data_file[0] || !tree_dir[0]);
}

inline void assert_arg(bool oops) {
    if (oops) {
        std::cerr << "Not enough input parameters!\n";
        exit(1);
    }
}

code parse_arg_short(const char* arg) {
    if (!std::strcmp(arg, "h")) return HELP;
    return ERROR;
}

code parse_arg_long(const char* arg) {
    if (!std::strcmp(arg, "help"))   return HELP;
    if (!std::strcmp(arg, "detail")) return DETAIL;
    return ERROR;
}

void print_help() {
    std::cout << "My Tree: \n";
    std::cout << "Usage:\n    mytree_test [options] data_file tree_dir\n";
    std::cout << "Arguments:\n";
    std::cout << "    data_file\n        file with testing data.\n";
    std::cout << "Options:\n";
    std::cout << "    --help, -h\n";
    std::cout << "        Show this help and exit.\n";
}
