# ifndef _MY_TYPEDEFS_H
# define _MY_TYPEDEFS_H

# define SIZEOF_LINE 1048576
# define SIZEOF_PATH 4096

typedef double        COMP_T;
// type of the value to be computed (parameters, training samples, etc)
// alternatives: double / float
typedef int           SUPV_T;
// type of the supervising information (classes, labels, etc)
// alternatives: any type of signed integer
typedef unsigned long N_DAT_T;
// type of the number of the data set
// alternatives: any type of integer
typedef unsigned int  DAT_DIM_T;
// type of the dimension of the data
// alternatives: any type of integer


# endif
