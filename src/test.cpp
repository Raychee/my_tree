#include <map>
#include <set>
#include <iostream>
#include <ctime>

# include "Histogram.hpp"

using namespace std;

int main(int argc, const char** argv) {

    unsigned int* a = new unsigned int[10];
    unsigned int* b = new unsigned int[10];
    a[0] = 1;
    a[1] = 1;
    a[2] = 9;
    a[3] = 9;
    a[4] = 9;
    a[5] = 9;
    a[6] = 9;
    a[7] = 9;
    a[8] = 1;
    a[9] = 1;
    memcpy(b, a, sizeof(int) * 10);
    Histogram<int, unsigned int> ha(a, 10);
    Histogram<unsigned int, int> hb(b, 10);

    cout << ha << endl;
    cout << hb << endl;

    // const int MaxVal = 30000000;
    // time_t startTime, endTime;
    // std::map<int, int> is;
    // std::map<int, int>::iterator isIter;

    // // no hint

    // std::cout << "start - no hint" << std::endl;
    // time(&startTime);

    // for (int i = 0; i < MaxVal; ++i)
    // is.insert(std::map<int, int>::value_type(i, i));

    // time(&endTime);
    // std::cout << "finish: " << difftime(endTime,startTime) << " seconds"
    // << std::endl;

    // // insert after hint

    // is.clear();
    // std::cout << "start - insert after hint" << std::endl;
    // time(&startTime);

    // isIter = (is.insert(std::map<int, int>::value_type(0, 0))).first;
    // for (int i = 1; i < MaxVal; ++i)
    // isIter = is.insert(isIter,std::map<int, int>::value_type(i, i));

    // time(&endTime);
    // std::cout << "finish: " << difftime(endTime,startTime) << " seconds"
    // << std::endl;

    // // insert before hint

    // is.clear();
    // std::cout << "start - insert before hint" << std::endl;
    // time(&startTime);

    // isIter = (is.insert(std::map<int, int>::value_type(MaxVal, MaxVal))).first;
    // for (int i = MaxVal-1; i >= 0; --i)
    // isIter = is.insert(isIter,std::map<int, int>::value_type(i, i));

    // time(&endTime);
    // std::cout << "finish: " << difftime(endTime,startTime) << " seconds"
    // << std::endl;

    // int x[MaxVal];
    // for (int i = 0; i < MaxVal; ++i) {
    //     x[i] = i;
    // }

    // std::cout << "start" << std::endl;
    // time(&startTime);

    // std::set<int> is(x, x + MaxVal);

    // time(&endTime);
    // std::cout << "finish: " << difftime(endTime,startTime) << " seconds" << std::endl;

    return 0;
}