# ifndef _HISTOGRAM_HPP
# define _HISTOGRAM_HPP

# include <iostream>
# include <map>


/// A wrapper class for holding a histogram.
/// 
/// A histogram can be implemented by an array or a std::map. If the 
/// histogram is sparse, the std::map would be the better implementation.
/// Otherwise an array is more efficient. The class would automatically 
/// choose its implementation during construction according to different 
/// inputs.
template<typename _INDEX, typename _VALUE>
class Histogram {
public:
  Histogram();
  Histogram(Histogram& some);
  /// Constructor of a histogram.
  /// 
  /// @param[in] _array An array of a histogram. If sparse enough, it will 
  ///                   be converted into a std::map and the array will be 
  ///                   released. Otherwise it will be directly encapsulated 
  ///                   into the class.
  Histogram(_VALUE* _array, _INDEX _length);
  ~Histogram();

  Histogram& operator=(Histogram& some);

  /// Insert or replace the histogram inside the class
  Histogram& insert(_VALUE* _array, _INDEX _length);
  /// Make a histogram of length "_bin" by counting the number of times that 
  /// the numbers 1~"_bin" appear in the array of length "_length".
  Histogram& stat(_INDEX* _array, _VALUE _length, _INDEX _bin = 0);
  /// Read-only access to an arbitary element of index i.
  _VALUE     operator[](_INDEX i);
  _INDEX     length() const { return length_; }
  Histogram& clear();

private:
    bool   alloc;
    bool   use_map;
    _INDEX length_;
    union histogram {
        _VALUE*                   array;
        std::map<_INDEX, _VALUE>* map;
    } hist;
};


template<typename _INDEX, typename _VALUE> inline 
Histogram<_INDEX, _VALUE>::Histogram():
                           alloc(false),
                           use_map(false),
                           length_(0) {
    hist.array = NULL;
}

template<typename _INDEX, typename _VALUE> inline 
Histogram<_INDEX, _VALUE>::Histogram(Histogram& some):
                           alloc(false),
                           use_map(some.use_map),
                           length_(some.length_) {
    if (use_map) hist.map = some.hist.map;
    else hist.array = some.hist.array;
}

template<typename _INDEX, typename _VALUE> inline 
Histogram<_INDEX, _VALUE>::Histogram(_VALUE* _array, _INDEX _length):
                           alloc(false),
                           use_map(false),
                           length_(0) {
    hist.array = NULL;
    insert(_array, _length);
}

template<typename _INDEX, typename _VALUE> inline 
Histogram<_INDEX, _VALUE>::~Histogram() {
    clear();
}

template<typename _INDEX, typename _VALUE> inline 
Histogram<_INDEX, _VALUE>& Histogram<_INDEX, _VALUE>::
operator=(Histogram& some) {
    if (use_map == some.use_map) {
        if ((!use_map && hist.array == some.hist.array)
            || (use_map && hist.map == some.hist.map)) return *this;
    }
    clear();
    use_map = some.use_map;
    length_ = some.length_;
    if (use_map) hist.map = some.hist.map;
    else hist.array = some.hist.array;
    return *this;
}

template<typename _INDEX, typename _VALUE>
Histogram<_INDEX, _VALUE>& Histogram<_INDEX, _VALUE>::
insert(_VALUE* _array, _INDEX  _length) {
    clear();
    _INDEX n_nonzero = 0, i_nonzero;
    for (_INDEX i = 0; i < _length; ++i) {
        if (_array[i]) { n_nonzero++; i_nonzero = i; }
    }
    use_map = _length / n_nonzero > 3;
    if (use_map) {
        hist.map = new std::map<_INDEX, _VALUE>;
        typename std::map<_INDEX, _VALUE>::iterator hist_it;
        hist_it = hist.map->insert(typename std::map<_INDEX, _VALUE>::
                            value_type(i_nonzero, _array[i_nonzero])).first;
        for (_INDEX i = i_nonzero - 1; i != 0; --i) {
            if (_array[i]) {
                hist_it = hist.map->insert(hist_it, typename std::map<_INDEX, _VALUE>::
                                    value_type(i, _array[i]));
            }
        }
        if (_array[0]) {
            hist_it = hist.map->insert(hist_it, typename std::map<_INDEX, _VALUE>::
                                value_type(0, _array[0]));
        }
        delete[] _array;
    }
    else {
        hist.array = _array;
    }
    alloc   = true;
    length_ = _length;
    return *this;
}

template<typename _INDEX, typename _VALUE>
Histogram<_INDEX, _VALUE>& Histogram<_INDEX, _VALUE>::
stat(_INDEX* _array, _VALUE  _length, _INDEX  _bin) {
    clear();
    if (!_bin) {
        for (_VALUE i = 0; i < _length; ++i) {
            if (_array[i] > _bin) _bin = _array[i];
        }
    }
    _VALUE* array = new _VALUE[_bin];
    for (_VALUE i = 0; i < _length; ++i) {
        if (_array[i]) ++array[_array[i] - 1];
    }
    insert(array, _bin);
    return *this;
}

template<typename _INDEX, typename _VALUE> inline 
_VALUE Histogram<_INDEX, _VALUE>::
operator[](_INDEX i) {
    if (use_map) {
        typename std::map<_INDEX, _VALUE>::iterator it = hist.map->find(i);
        if (it != hist.map->end()) return it->second;
        else return 0;
    }
    else {
        return hist.array[i];
    }
}

template<typename _INDEX, typename _VALUE> inline 
Histogram<_INDEX, _VALUE>& Histogram<_INDEX, _VALUE>::
clear() {
    if (alloc) {
        if (use_map) delete hist.map;
        else delete[] hist.array;
        alloc = false;
    }
    use_map    = false;
    length_    = 0;
    hist.array = NULL;
    return *this;
}

template<typename _INDEX, typename _VALUE>
std::ostream& operator<<(std::ostream& out, Histogram<_INDEX, _VALUE>& h) {
    out << h[0];
    _INDEX length = h.length();
    for (_INDEX i = 1; i < length; ++i) {
        out << " " << h[i];
    }
    return out;
}

# endif
