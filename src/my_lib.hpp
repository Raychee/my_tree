# ifndef _MY_LIB_HPP
# define _MY_LIB_HPP

template <typename _TYPE> _TYPE strto(const char* str);
template <typename _TYPE> _TYPE strto(const char* str, char*& str_end);
template <typename _TYPE> _TYPE strto(const char* str, char*& str_end, int base);

char* strtostr(char* str);

# endif
