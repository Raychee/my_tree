# ifndef _SGD_H
# define _SGD_H

# include <iostream>
# include <cstring>
# include <limits>
// # include <forward_list>

# include "LabelStat.hpp"

/// General Gradient Descent solver model
/// 
/// @tparam _COMP_T    type of the value to be computed (parameters, data samples, etc)
/// @tparam _SUPV_T    type of the supervising information (classes, labels, etc)
/// @tparam _DAT_DIM_T type of the dimension of the data
/// @tparam _N_DAT_T   type of the number of the data set
/// 
template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
class GD {
public:
    class GDParam {
    public:
        GDParam(_DAT_DIM_T   _dimension            = 0,
                char         _verbosity            = 1,
                _COMP_T      _eta0                 = 0,
                unsigned int _n_iter               = 200,
                _COMP_T      _err                  = 1e-8,
                _N_DAT_T     _eta0_try_subsample   = 0,
                _COMP_T      _eta0_try_1st         = 0.1,
                _COMP_T      _eta0_try_factor      = 3,
                bool         _show_obj_each_iter   = false);
        ~GDParam() {}

        // some interfaces to write private variables in this class
        GDParam& verbosity(char _verbosity)
            { v = _verbosity; return *this; }
        GDParam& dimension(_DAT_DIM_T _dimension) {
            if (d) {
                std::cerr << "WARNING: SGD: "
                          << "Dimensionality has already been specified. "
                          << "Nothing is changed." << std::endl;
            }
            else d = _dimension;
            return * this;
        }
        GDParam& init_learning_rate(_COMP_T _eta0)
            { eta0 = _eta0; return *this; }
        GDParam& num_of_iterations(unsigned int _n_iter)
            { n_iter = _n_iter; return *this; }
        GDParam& accuracy(_COMP_T _err)
            { err = _err; return *this; }
        GDParam& num_of_subsamples(_N_DAT_T _eta0_try_subsample)
            { eta0_try_subsample = _eta0_try_subsample; return *this; }
        GDParam& learning_rate_1st_try(_COMP_T _eta0_try_1st)
            { eta0_try_1st = _eta0_try_1st; return *this; }
        GDParam& learning_rate_try_factor(_COMP_T _eta0_try_factor)
            { eta0_try_factor = _eta0_try_factor; return *this; }
        GDParam& show_obj_each_iteration(bool _compute_obj)
            { show_obj_each_iter = _compute_obj; return *this; }
        GDParam& ostream_of_training_process(std::ostream& _out_training_proc)
            { out_training_proc = &_out_training_proc; return *this; }

        // some interfaces to read private variables in this class
        char          verbosity()                   const { return v; }
        _DAT_DIM_T    dimension()                   const { return d; }
        _COMP_T       init_learning_rate()          const { return eta0; }
        unsigned int  num_of_iterations()           const { return n_iter; }
        _COMP_T       accuracy()                    const { return err; }
        _N_DAT_T      num_of_subsamples()           const { return eta0_try_subsample; }
        _COMP_T       learning_rate_1st_try()       const { return eta0_try_1st; }
        _COMP_T       learning_rate_try_factor()    const { return eta0_try_factor; }
        bool          show_obj_each_iteration()     const { return show_obj_each_iter; }
        std::ostream* ostream_of_training_process() const { return out_training_proc; }

        GDParam&      ostream_this(std::ostream& out);

    private:
        _DAT_DIM_T   d;             ///< dimensionality
        char         v;             ///< verbosity
        _COMP_T      eta0;          ///< the initial learning rate
        /// Stopping criteria 1: max number of iterations
        unsigned int n_iter;
        /// Stopping criteria 2: residual error upper bound
        _COMP_T      err;
        /// the ratio of sampling from the data set for determining eta0
        _N_DAT_T     eta0_try_subsample;
        /// the first guess of eta0 (if eta0 is not specified by user)
        _COMP_T      eta0_try_1st;
        /// the factor that eta0 multiplies for each try
        _COMP_T      eta0_try_factor;
        /// whether to compute the objective value after each iteration
        bool         show_obj_each_iter;
        // pointer to the ostream which is used for the output of parameters.
        // If NULL, then nothing will be output
        std::ostream* out_training_proc;
    };

    GD(_DAT_DIM_T   _dimension            = 0,
       char         _verbosity            = 1,
       _COMP_T      _eta0                 = 0,
       unsigned int _n_iter               = 200,
       _COMP_T      _err                  = 1e-8,
       _N_DAT_T     _eta0_try_subsample   = 0,
       _COMP_T      _eta0_try_1st         = 0.1,
       _COMP_T      _eta0_try_factor      = 3,
       bool         _show_obj_each_iter   = false);
    GD(GDParam& _gd_param);
    GD(GD& some);
    GD& operator=(GD& some);
    virtual ~GD();

    /// Feed the training data into the solver.
    /// 
    /// @param[in] _data The matrix that holds all the training data.
    /// @param[in] _n    Number of samples in the matrix.
    /// @param[in] _y    The array containing labels of each sample.
    /// @param[in] _x    If only a subset of data is used as training data, "x" is 
    ///                  an index array of length "s_x" that holds all the indexes
    ///                  of training samples. "x==NULL" if all the data are for 
    ///                  training.
    /// @param[in] _s_x  Number of training samples. Ignored if "x==NULL".
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    training_data(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y,
                  _N_DAT_T* _x = NULL, _N_DAT_T _s_x = 0);

    /// Feed the training data into the solver.
    /// 
    /// This function call is proper only when "data", "n", and "y" have been 
    /// indicated.
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    training_data(_N_DAT_T* _x, _N_DAT_T _s_x);

    /// Train the parameters with the explicit training data. 
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y,
          _N_DAT_T* _x = NULL, _N_DAT_T _s_x = 0);

    /// Train the parameters with the explicit training data. 
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train(_N_DAT_T* _x, _N_DAT_T _s_x);

    /// Train the parameters with the data fed beforehand.
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train();

    /// Test all the "n" input data of dimension "dim" and output the result in 
    /// the corresponding position of the array "y".
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    test(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y,
         _N_DAT_T* _x = NULL, _N_DAT_T _s_x = 0);

    /// Test one data sample.
    virtual _SUPV_T test_one(_COMP_T* dat_i, _DAT_DIM_T d) const = 0;

    /// Output the SGD solver to a standard ostream.
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    ostream_this(std::ostream& out);

    /// Output all the training parameters to a standard ostream.
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    ostream_param(std::ostream& out) = 0;

protected:
    _COMP_T*   data;     ///< data buffer (external)
    _N_DAT_T   n;        ///< total number of samples in the buffer
    _SUPV_T*   y;        ///< labels (external)
    
    // buffer variable for efficient training
    LabelStat<_SUPV_T, _N_DAT_T> stat;

    GDParam*   gd_param; ///< gradient descent parameters (external)
    // temporary parameters during learning
    _N_DAT_T   t;        ///< current pass (number of data samples passed)
    _COMP_T    eta;      ///< current learning rate

    /// Determine the initial learning rate eta0 automatically according to the 
    /// given data set.
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    try_learning_rate();

    virtual _COMP_T compute_learning_rate(_COMP_T eta0, _N_DAT_T t) = 0;

    /// Train with all the data in "data".
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
    train_batch(_COMP_T*   _data,
                _DAT_DIM_T _d,
                _N_DAT_T   _n,
                _SUPV_T*   _y) = 0;

    /// Train with a subset of data indicated by "x".
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
    train_batch(_COMP_T*   _data,
                _DAT_DIM_T _d,
                _N_DAT_T*  _x,
                _N_DAT_T   _n,
                _SUPV_T*   _y) = 0;

    /// Compute the object function (e.g. sum of empirical losses plus a 
    /// regularizing term) given the current parameters and data.
    virtual _COMP_T compute_obj(_COMP_T*   _data,
                                _DAT_DIM_T _d,
                                _N_DAT_T   _n,
                                _SUPV_T*   _y) = 0;
    virtual _COMP_T compute_obj(_COMP_T*   _data,
                                _DAT_DIM_T _d,
                                _N_DAT_T*  _x,
                                _N_DAT_T   _n,
                                _SUPV_T*   _y) = 0;

private:
    /// Mark whether the param structure that "gd_param" points to is allocated 
    /// by the constructor during a GD object creation.
    bool alloc_gd_param;
    /// Create and return a temporary duplicate of the current GD solver. Used 
    /// in determin_eta0().
    virtual GD* duplicate_this() = 0;
};


/// General Stochastic Gradient Descent solver model
/// 
/// @tparam _COMP_T    type of the value to be computed (parameters, data samples, etc)
/// @tparam _SUPV_T    type of the supervising information (classes, labels, etc)
/// @tparam _DAT_DIM_T type of the dimension of the data
/// @tparam _N_DAT_T   type of the number of the data set
/// 
template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
class SGD : public GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T> {
public:
    /// Parameter structure for a general Stochastic Gradient Descent solver model
    class SGDParam {
    public:
        SGDParam(_N_DAT_T     _s_batch             = 1,
                 unsigned int _n_epoch             = 200);
        ~SGDParam() {}

        // some interfaces to write private variables in this class
        SGDParam& size_of_batches(_N_DAT_T _s_batch)
            { s_batch = _s_batch; return *this; }
        SGDParam& num_of_epoches(unsigned int _n_epoch)
            { n_epoch = _n_epoch; return *this; }

        // some interfaces to read private variables in this class
        _N_DAT_T      size_of_batches()     const { return s_batch; }
        unsigned int  num_of_epoches()      const { return n_epoch; }

        SGDParam&     ostream_this(std::ostream& out);

    private:
        _N_DAT_T      s_batch;       ///< size of mini-batches. 1 is completely SGD.
        unsigned int  n_epoch;       ///< number of epoches
    };//class SGDParam

    SGD(_DAT_DIM_T   _dimension            = 0,
        char         _verbosity            = 1,
        _COMP_T      _eta0                 = 0,
        _N_DAT_T     _s_batch              = 1,
        unsigned int _n_epoch              = 200,
        unsigned int _n_iter               = 200,
        _COMP_T      _err                  = 0,
        _N_DAT_T     _eta0_try_subsample   = 0,
        _COMP_T      _eta0_try_1st         = 0.1,
        _COMP_T      _eta0_try_factor      = 3,
        bool         _show_obj_each_iter   = false);
    SGD(typename GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& _gd_param,
        _N_DAT_T     _s_batch = 1,
        unsigned int _n_epoch = 200);
    SGD(typename GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& _gd_param,
        SGDParam& _sgd_param);
    SGD(SGD& some);
    virtual ~SGD();

    /// Train the parameters with ALL the "n" input data of dimension "dim" and 
    /// the corresponding supervising data "y". 
    /// 
    /// The order of the data samples are automatically shuffled before each 
    /// epoch. Every parameter is applied with update rule once for each input 
    /// data sample.
    virtual SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train();

    SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    rand_index(_N_DAT_T* x, _N_DAT_T n);

    /// Output the SGD solver to a standard ostream.
    virtual SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    ostream_this(std::ostream& out);

protected:
    SGDParam*   sgd_param;        ///< sgd parameters (external)

    /// Train the parameters "param" with "m" data samples in "dat" whose indexes
    /// are specified by "dat_idx". 
    /// 
    /// The training precedure follows exactly the order of indexes in "dat_idx". 
    /// The data pass only once (1-epoch). Every parameter is applied with 
    /// update rule once for each input data sample. "y" should be the WHOLE 
    /// supervising data ("n", same length as "dat"), not just length of "m".
    virtual SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train_epoch(_DAT_DIM_T _d, _N_DAT_T* _x,
                _N_DAT_T _s_batch,
                _N_DAT_T _n_batch,
                _N_DAT_T _n_remain,
                char _v);

private:
    /// Mark whether the param structure that "sgd_param" points to is allocated 
    /// by the constructor during a SGD object creation.
    bool alloc_sgd_param;
};

/// A wrapper of function ostream_this(): operator << overload for convenience.
template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline std::ostream& operator<<(std::ostream& out,
                                GD<_COMP_T,
                                   _SUPV_T,
                                   _DAT_DIM_T,
                                   _N_DAT_T>& gd) {
    gd.ostream_this(out);
    return out;
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
GDParam::GDParam(_DAT_DIM_T   _dimension,
                 char         _verbosity,
                 _COMP_T      _eta0,
                 unsigned int _n_iter,
                 _COMP_T      _err,
                 _N_DAT_T     _eta0_try_subsample,
                 _COMP_T      _eta0_try_1st,
                 _COMP_T      _eta0_try_factor,
                 bool         _show_obj_each_iter):
         d(_dimension),
         v(_verbosity),
         eta0(_eta0),
         n_iter(_n_iter),
         err(_err),
         eta0_try_subsample(_eta0_try_subsample),
         eta0_try_1st(_eta0_try_1st),
         eta0_try_factor(_eta0_try_factor),
         show_obj_each_iter(_show_obj_each_iter),
         out_training_proc(NULL) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline typename GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam::
ostream_this(std::ostream& out) {
    out << "Gradient Descent parameters:\n"
        << "\tVerbosity = " << (int)v << "\n"
        << "\tDimension = " << d << "\n"
        << "\tMax number of iterations = " << n_iter << "\n"
        << "\tStopping accuracy = " << err;
    if (eta0)
        out << "\n\tInitial learning rate = " << eta0;
    return *this;
}


template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
GD(_DAT_DIM_T   _dimension,
   char         _verbosity,
   _COMP_T      _eta0,
   unsigned int _n_iter,
   _COMP_T      _err,
   _N_DAT_T     _eta0_try_subsample,
   _COMP_T      _eta0_try_1st,
   _COMP_T      _eta0_try_factor,
   bool         _show_obj_each_iter)
  :data(NULL),
   n(0),
   y(NULL),
   t(0),
   eta(_eta0),
   alloc_gd_param(true) {
    gd_param = new GDParam(_dimension,
                           _verbosity,
                           _eta0,
                           _n_iter,
                           _err,
                           _eta0_try_subsample,
                           _eta0_try_1st,
                           _eta0_try_factor,
                           _show_obj_each_iter);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
GD(GDParam& _gd_param)
  :data(NULL),
   n(0),
   y(NULL),
   gd_param(&_gd_param),
   t(0),
   alloc_gd_param(false) {
    eta = gd_param->init_learning_rate();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
GD(GD& some)
  :data(some.data),
   n(some.n),
   y(some.y),
   stat(some.stat),
   gd_param(some.gd_param),
   t(some.t),
   eta(some.eta),
   alloc_gd_param(false) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
operator=(GD& some) {
    if (alloc_gd_param && gd_param != some.gd_param) {
        delete gd_param;
        alloc_gd_param = false;
    }
    data           = some.data;
    n              = some.n;
    y              = some.y;
    stat           = some.stat;
    gd_param       = some.gd_param;
    t              = some.t;
    eta            = some.eta;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
~GD() {
    if (alloc_gd_param) delete gd_param;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
training_data(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y,
              _N_DAT_T* _x, _N_DAT_T _s_x) {
    data = _data;
    n    = _n;
    y    = _y;
    if (_x && _s_x) stat.stat(y, _s_x, _x);
    else stat.stat(y, n);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
training_data(_N_DAT_T* _x, _N_DAT_T _s_x) {
    stat.stat(y, _s_x, _x);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y, _N_DAT_T* _x, _N_DAT_T _s_x) {
    training_data(_data, _n, _y, _x, _s_x);
    return train();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train(_N_DAT_T* _x, _N_DAT_T _s_x) {
    training_data(_x, _s_x);
    return train();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train() {
    _DAT_DIM_T   dim                = gd_param->dimension();
    char         verbosity          = gd_param->verbosity();
    _COMP_T      eta0               = gd_param->init_learning_rate();
    unsigned int n_iter             = gd_param->num_of_iterations();
    _COMP_T      err                = gd_param->accuracy();
    bool         show_obj_each_iter = gd_param->show_obj_each_iteration();
    _N_DAT_T     n_sample           = stat.num_of_samples();
    _N_DAT_T*    x                  = NULL;
    if (!dim) {
        std::cerr << "WARNING: GD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    if (verbosity >= 1) {
        std::cout << "GD Training: \n    Data: " << n_sample << " samples, "
                  << dim << " feature dimensions.\n    Stopping Criterion: "
                  << n_iter << " iterations";
        if (err > 0) std::cout << " or accuracy higher than " << err;
        std::cout <<  ".\nGD Training: begin." << std::endl;
    }
    if (!eta0) try_learning_rate();
    _COMP_T obj0 = std::numeric_limits<_COMP_T>::max();
    _COMP_T obj1;
    if (n > n_sample) {
        x = new _N_DAT_T[n_sample];
        stat.index_of_samples(x);
    }
    if (verbosity >= 1) {
        std::cout << "Training ... ";
        if (verbosity > 1) std::cout << std::endl;
        else std::cout.flush();
    }
    for (t = 0; t < n_iter; ++t) {
        if (verbosity >= 2) {
            std::cout << "    Iteration " << t+1 << " ... " 
                      << std::flush;
        }
        eta = compute_learning_rate(eta0, t);
        if (x) {
            train_batch(data, dim, x, n_sample, y);
            obj1 = compute_obj(data, dim, x, n_sample, y);
        }
        else {
            train_batch(data, dim, n, y);
            obj1 = compute_obj(data, dim, n, y);
        }
        if (verbosity >= 2) {
            std::cout << "Done. eta = " << eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = " << obj1 << ".";
            }
            std::cout << "." << std::endl;
        }
        // DEBUG
        std::ostream* out = gd_param->ostream_of_training_process();
        if (out) {
            ostream_param(*out);
            *out << " ";
            *out << obj1 << std::endl;
        }
        if (err > 0 && obj0 - obj1 < err) break;
        obj0 = obj1;
    }
    if (verbosity >= 1) {
        if (verbosity == 1) std::cout << "Done.\n";
        std::cout << "GD Training: finished. \n";
        if (t < n_iter)
            std::cout << "    Training stopped at iteration " << t + 1 
                      << " with convergence.";
        else
            std::cout << "    Max number of iterations has been reached.";
        std::cout << std::endl;
    }
    delete[] x;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
test(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y, _N_DAT_T* _x, _N_DAT_T _s_x) {
    _DAT_DIM_T dim       = gd_param->dimension();
    char       verbosity = gd_param->verbosity();    
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    if (verbosity >= 1) {
        std::cout << "GD Testing: \n  Data: " << _n << " samples.\nTesting ... ";
        if (verbosity > 1) std::cout << std::endl;
        else std::cout.flush();
    }
    if (_x && _s_x) {
        for (_N_DAT_T i = 0; i < _s_x; ++i) {
            _COMP_T* dat_i = _data + dim * _x[i];
            if (verbosity >= 2)
                std::cout << "  Sample " << i << " ... " << std::flush;
            _y[i] = test_one(dat_i, dim);
            if (verbosity >= 2)
                std::cout << "Done. Predicted label: " << _y[i] << "."
                          << std::endl;
        }
    }
    else {
        _COMP_T* dat_i = _data;
        for (_N_DAT_T i = 0; i < _n; ++i, dat_i += dim) {
            if (verbosity >= 2)
                std::cout << "  Sample " << i << " ... " << std::flush;
            _y[i] = test_one(dat_i, dim);
            if (verbosity >= 2)
                std::cout << "Done. Predicted label: " << _y[i] << "."
                          << std::endl;
        }
    }
    if (verbosity == 1) std::cout << "Done.\n";
    if (verbosity >= 1) std::cout << "GD Testing: finished." << std::endl;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
ostream_this(std::ostream& out) {
    out << "Data: " << stat.num_of_samples() << " samples, "
        << stat.num_of_labels() << " classes.\n";
    gd_param->ostream_this(out);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
try_learning_rate() {
    _DAT_DIM_T dim = gd_param->dimension();
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }

    _COMP_T eta0_try1, eta0_try2, obj_try1, obj_try2;
    char verbosity = gd_param->verbosity();
    if (verbosity >= 1) {
        std::cout << "Initializing ... ";
        if (verbosity >= 2) {
            std::cout << "\n    No initial learning rate \"eta0\" specified. "
                      << "Deciding automatically ... ";
            if (verbosity > 2) std::cout << std::endl;
            else std::cout.flush();
        }
        std::cout.flush();
    }

    _N_DAT_T n_subsample = gd_param->num_of_subsamples();
    if (!n_subsample) n_subsample = stat.num_of_samples();
    _N_DAT_T* sub_x_i = new _N_DAT_T[n_subsample];
    stat.rand_index(sub_x_i, n_subsample);

    LabelStat<_SUPV_T, _N_DAT_T> stat_new(y, n_subsample, sub_x_i);

    eta0_try1 = gd_param->learning_rate_1st_try();
    if (verbosity >= 3)
        std::cout << "        Trying eta0 = " << eta0_try1 << " ... " << std::flush;
    GD* tempGD = duplicate_this();
    tempGD->eta = eta0_try1;
    tempGD->stat = stat_new;
    tempGD->train_batch(data, dim, sub_x_i, n_subsample, y);
    obj_try1 = tempGD->compute_obj(data, dim, sub_x_i, n_subsample, y);
    delete tempGD;
    if (verbosity >= 3)
        std::cout << "Done. Obj = " << obj_try1 << "." << std::endl;
    _COMP_T eta0_try_factor = gd_param->learning_rate_try_factor();
    eta0_try2 = eta0_try1 * eta0_try_factor;
    if (verbosity >= 3)
        std::cout << "        Trying eta0 = " << eta0_try2 << "... " << std::flush;
    tempGD = duplicate_this();
    tempGD->eta = eta0_try2;
    tempGD->stat = stat_new;
    tempGD->train_batch(data, dim, sub_x_i, n_subsample, y);
    obj_try2 = tempGD->compute_obj(data, dim, sub_x_i, n_subsample, y);
    delete tempGD;
    if (verbosity >= 3)
        std::cout << "Done. Obj = " << obj_try2 << "." << std::endl;
    if (obj_try1 < obj_try2) {
        eta0_try_factor = 1 / eta0_try_factor;
        obj_try2        = obj_try1;
        eta0_try2       = eta0_try1;
    }
    do {
        obj_try1  = obj_try2;
        eta0_try1 = eta0_try2;
        eta0_try2 = eta0_try1 * eta0_try_factor;
        if (verbosity >= 3)
            std::cout << "        Trying eta0 = " << eta0_try2 << "... " << std::flush;
        tempGD = duplicate_this();
        tempGD->eta = eta0_try2;
        tempGD->stat = stat_new;
        tempGD->train_batch(data, dim, sub_x_i, n_subsample, y);
        obj_try2 = tempGD->compute_obj(data, dim, sub_x_i, n_subsample, y);
        delete tempGD;
        if (verbosity >= 3)
            std::cout << "Done. Obj = " << obj_try2 << "." << std::endl;
    } while (obj_try1 > obj_try2);
    gd_param->init_learning_rate(eta0_try1 * 0.9);
    if (verbosity == 1) std::cout << "Done." << std::endl;
    if (verbosity >= 2) {
        if (verbosity == 2) std::cout << "Done.\n";
        std::cout << "    Setting eta0 = " << eta0_try1 * 0.9 << "." << std::endl;
    }

    delete[] sub_x_i;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGDParam::SGDParam(_N_DAT_T     _s_batch,
                   unsigned int _n_epoch):
          s_batch(_s_batch),
          n_epoch(_n_epoch) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline typename SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::SGDParam&
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::SGDParam::
ostream_this(std::ostream& out) {
    out << "Stochastic Gradient Descent parameters:\n"
        << "\tMax number of epoches = " << n_epoch;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGD(_DAT_DIM_T   _dimension,
    char         _verbosity,
    _COMP_T      _eta0,
    _N_DAT_T     _s_batch,
    unsigned int _n_epoch,
    unsigned int _n_iter,
    _COMP_T      _err,
    _N_DAT_T     _eta0_try_subsample,
    _COMP_T      _eta0_try_1st,
    _COMP_T      _eta0_try_factor,
    bool         _show_obj_each_iter)
   :GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>(_dimension,
                                               _verbosity,
                                               _eta0,
                                               _n_iter,
                                               _err,
                                               _eta0_try_subsample,
                                               _eta0_try_1st,
                                               _eta0_try_factor,
                                               _show_obj_each_iter),
    alloc_sgd_param(true) {
    sgd_param = new SGDParam(_s_batch,_n_epoch);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGD(typename GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& _gd_param,
    _N_DAT_T     _s_batch,
    unsigned int _n_epoch)
   :GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>(_gd_param),
    alloc_sgd_param(true) {
    sgd_param = new SGDParam(_s_batch,_n_epoch);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGD(typename GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& _gd_param,
    SGDParam& _sgd_param)
   :GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>(_gd_param),
    sgd_param(&_sgd_param),
    alloc_sgd_param(false) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGD(SGD& some)
   :GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>(some),
    sgd_param(some.sgd_param),
    alloc_sgd_param(false) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
~SGD() {
    if (alloc_sgd_param) delete sgd_param;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train() {
    _DAT_DIM_T   dim                = this->gd_param->dimension();
    char         verbosity          = this->gd_param->verbosity();
    _COMP_T      err                = this->gd_param->accuracy();
    bool         show_obj_each_iter = this->gd_param->show_obj_each_iteration();
    _N_DAT_T     s_batch            = sgd_param->size_of_batches();
    unsigned int n_epoch            = sgd_param->num_of_epoches();
    _N_DAT_T     n_sample           = this->stat.num_of_samples();
    _N_DAT_T*    rand_x;
    _N_DAT_T     n_batch  = n_sample / s_batch;
    _N_DAT_T     n_remain = n_sample - s_batch * n_batch;
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    if (verbosity >= 1) {
        std::cout << "SGD Training: \n    Data: " << n_sample << " samples, "
                  << dim << " feature dimensions. " << s_batch 
                  << " samples per batch.\n    Stopping Criterion: "
                  << n_epoch << " epoches";
        if (err > 0) std::cout << " or accuracy higher than " << err;
        std::cout <<  ".\nSGD Training: begin." << std::endl;
    }
    if (!this->gd_param->init_learning_rate()) this->try_learning_rate();
    this->t = 0;
    rand_x = new _N_DAT_T[n_sample];
    _COMP_T obj0 = std::numeric_limits<_COMP_T>::max();
    _COMP_T obj1;
    unsigned int i;
    if (verbosity >= 1) {
        std::cout << "Training ... ";
        if (verbosity > 1) std::cout << std::endl;
        else std::cout.flush();
    }
    for (i = 0; i < n_epoch; ++i) {
        if (verbosity >= 3)
            std::cout << "    Shuffling the data set... " << std::flush;
        this->stat.rand_index(rand_x);
        if (verbosity >= 3) std::cout << "Done." << std::endl; 
        if (verbosity >= 2) {
            std::cout << "    Epoch " << i + 1  << " ... ";
            if (verbosity >= 3) std::cout << std::endl;
            else std::cout.flush();
        }
        train_epoch(dim, rand_x, s_batch, n_batch, n_remain, verbosity);
        obj1 = this->compute_obj(this->data, dim, rand_x, n_sample, this->y);
        if (verbosity == 2) {
            std::cout << "Done. eta = " << this->eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = " << obj1;
            }
            std::cout << "." << std::endl;
        }
        // DEBUG
        std::ostream* out = this->gd_param->ostream_of_training_process();
        if (out) {
            this->ostream_param(*out);
            *out << " ";
            *out << obj1 << std::endl;
        }
        if (err > 0 && obj0 - obj1 < err) break;
        obj0 = obj1;
    }

    if (verbosity >= 1) {
        if (verbosity == 1) std::cout << "Done.\n";
        std::cout << "SGD Training: finished.\n";
        if (i < n_epoch)
            std::cout << "    Training stopped at epoch " << i + 1 
                      << " with convergence.";
        else 
            std::cout << "    Max number of epoches has been reached.";
        std::cout << std::endl;
    }
    delete[] rand_x;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
rand_index(_N_DAT_T* x, _N_DAT_T n) {
    for (_N_DAT_T i = 0; i < n; ++i) {
        _N_DAT_T temp_i = std::rand() % (n - i) + i;
        _N_DAT_T temp   = x[temp_i];
        x[temp_i]       = x[i];
        x[i]            = temp;
    }
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
inline SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
ostream_this(std::ostream& out) {
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::ostream_this(out);
    out << "\n";
    sgd_param->ostream_this(out);
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train_epoch(_DAT_DIM_T _d, _N_DAT_T* _x,
            _N_DAT_T _s_batch,
            _N_DAT_T _n_batch,
            _N_DAT_T _n_remain,
            char _v) {
    _COMP_T   eta0               = this->gd_param->init_learning_rate();
    bool      show_obj_each_iter = this->gd_param->show_obj_each_iteration();
    _N_DAT_T* x_batch            = _x;
    _N_DAT_T  num_of_batches;
    if (_n_remain > 0) num_of_batches = _n_batch + 1;
    else num_of_batches = _n_batch;
    for (_N_DAT_T i = 0; i < _n_batch; ++i, ++this->t, x_batch += _s_batch) {
        if (_v >= 3) {
            std::cout << "        Batch " << i + 1 << " / " << num_of_batches 
                      << " ... " << std::flush;
        }
        this->eta = this->compute_learning_rate(eta0, this->t);
        this->train_batch(this->data, _d, x_batch, _s_batch, this->y);
        if (_v >= 3) {
            std::cout << "Done. eta = " << this->eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = "
                          << this->compute_obj(this->data, _d, _x,
                                               _s_batch * _n_batch + _n_remain,
                                               this->y);
            }
            std::cout << "." << std::endl;
        }
    }
    if (_n_remain > 0) {
        if (_v >= 3) {
            std::cout << "        Batch " << num_of_batches << " / " 
                      << num_of_batches << " ... " << std::flush;
        }
        this->eta = this->compute_learning_rate(eta0, this->t);
        this->train_batch(this->data, _d, x_batch, _n_remain, this->y);
        ++this->t;
        if (_v >= 3) {
            std::cout << "Done. eta = " << this->eta;
            if (show_obj_each_iter) {
                std::cout << ", Objective = "
                          << this->compute_obj(this->data, _d, _x,
                                               _s_batch * _n_batch + _n_remain,
                                               this->y);
            }
            std::cout << "." << std::endl;
        }
    }
    return *this;
}

# endif
