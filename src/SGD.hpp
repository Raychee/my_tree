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
                _COMP_T      _err                  = 1e-4,
                float        _eta0_try_sample_rate = 0.3,
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
        GDParam& learning_rate_sample_rate(float _eta0_try_sample_rate)
            { eta0_try_sample_rate = _eta0_try_sample_rate; return *this; }
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
        float         learning_rate_sample_rate()   const { return eta0_try_sample_rate; }
        _COMP_T       learning_rate_1st_try()       const { return eta0_try_1st; }
        _COMP_T       learning_rate_try_factor()    const { return eta0_try_factor; }
        bool          show_obj_each_iteration()  const { return show_obj_each_iter; }
        std::ostream* ostream_of_training_process() const { return out_training_proc; }
    private:
        _DAT_DIM_T   d;             ///< dimensionality
        char         v;             ///< verbosity
        _COMP_T      eta0;          ///< the initial learning rate
        /// Stopping criteria 1: max number of iterations
        unsigned int n_iter;
        /// Stopping criteria 2: residual error upper bound
        _COMP_T      err;
        /// the ratio of sampling from the data set for determining eta0
        float        eta0_try_sample_rate;
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
       _COMP_T      _err                  = 1e-4,
       float        _eta0_try_sample_rate = 0.3,
       _COMP_T      _eta0_try_1st         = 0.1,
       _COMP_T      _eta0_try_factor      = 3,
       bool         _show_obj_each_iter   = false);
    GD(GDParam& _gd_param);
    GD(GD& some);
    virtual ~GD();

    /// Feed the training data into the solver.
    /// 
    /// @param[in] _data The matrix that holds all the training data.
    /// @param[in] _n    Number of samples in the matrix.
    /// @param[in] _y    The array containing labels of each sample.
    /// @param[in] _x    If only a subset of data is used as training data, "x" is 
    ///                  an index array of length "s_x" that holds all the indexes
    ///                  of training samples.
    /// @param[in] _s_x  Number of training samples.
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    training_data(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y,
                  _N_DAT_T* _x = NULL, _N_DAT_T _s_x = 0);
    /// Train the parameters with ALL the "n" input data of dimension "dim" and 
    /// the corresponding supervising data "y". 
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y,
          _N_DAT_T* _x = NULL, _N_DAT_T _s_x = 0);
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train();

    /// Test all the "n" input data of dimension "dim" and output the result in 
    /// the corresponding position of the array "y".
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    test(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y,
         _N_DAT_T* _x = NULL, _N_DAT_T _s_x = 0);

    /// Compute the object function (e.g. sum of empirical losses plus a 
    /// regularizing term) given the current parameters and data.
    virtual _COMP_T compute_obj() = 0;

    /// Output the SGD solver to a standard ostream.
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    ostream_this(std::ostream& out) = 0;

    /// Output all the training parameters to a standard ostream.
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    ostream_param(std::ostream& out) = 0;

protected:
    _COMP_T*   data;       ///< data buffer
    _N_DAT_T   n;          ///< total number of samples in the buffer
    _SUPV_T*   y;          ///< labels
    
    // buffer variable for efficient training
    LabelStat<_SUPV_T, _N_DAT_T> stat;

    GDParam*  gd_param;
    // temporary parameters during learning
    _N_DAT_T t;            ///< current pass (number of data samples passed)
    _COMP_T  eta;          ///< current learning rate

    /// Determine the initial learning rate eta0 automatically according to the 
    /// given data set.
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    try_learning_rate();
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    compute_learning_rate(_COMP_T eta0) = 0;
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
    train_iteration(_N_DAT_T* x = NULL, _N_DAT_T s_x = 0) = 0;

    /// Test one data sample.
    virtual _SUPV_T test_one(_COMP_T* dat_i, _DAT_DIM_T d) const = 0;

private:
    /// Mark whether the param structure that "gd_param" points to is allocated 
    /// by the constructor during a GD object creation.
    bool alloc_gd_param;
    /// Create and return a temporary duplicate of the current GD solver. Used 
    /// in determin_eta0().
    virtual GD* get_temp_dup() = 0;
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
        SGDParam(unsigned int _n_epoch             = 5,
                 bool         _show_obj_each_epoch = false);
        ~SGDParam() {}

        // some interfaces to write private variables in this class
        SGDParam& num_of_epoches(unsigned int _n_epoch)
            { n_epoch = _n_epoch; return *this; }
        SGDParam& show_obj_each_epoch(bool _compute_obj)
            { show_obj_each_epoch_ = _compute_obj; return *this; }

        // some interfaces to read private variables in this class
        unsigned int  num_of_epoches()      const { return n_epoch; }
        bool          show_obj_each_epoch() const { return show_obj_each_epoch_; }

    private:
        unsigned int  n_epoch;       ///< number of epoches
        /// whether to compute the objective value after each epoch
        bool          show_obj_each_epoch_;
    };//class SGDParam

    SGD(_DAT_DIM_T   _dimension            = 0,
        char         _verbosity            = 1,
        _COMP_T      _eta0                 = 0,
        unsigned int _n_epoch              = 5,
        unsigned int _n_iter               = 200,
        _COMP_T      _err                  = 1e-4,
        float        _eta0_try_sample_rate = 0.3,
        _COMP_T      _eta0_try_1st         = 0.1,
        _COMP_T      _eta0_try_factor      = 3,
        bool         _show_obj_each_epoch  = false,
        bool         _show_obj_each_iter   = false);
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
    SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train();

protected:
    SGDParam*   sgd_param;        ///< All the necessary parameters

    /// Train the parameters "param" with "m" data samples in "dat" whose indexes
    /// are specified by "dat_idx". 
    /// 
    /// The training precedure follows exactly the order of indexes in "dat_idx". 
    /// The data pass only once (1-epoch). Every parameter is applied with 
    /// update rule once for each input data sample. "y" should be the WHOLE 
    /// supervising data ("n", same length as "dat"), not just length of "m".
    SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train_epoch(_N_DAT_T* x, _N_DAT_T s_x);

    /// Update EVERY parameter once with one input data 
    /// (Sub-routine of SGD::train_epoch).
    virtual SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train_one(_COMP_T* dat_i, _DAT_DIM_T d) = 0;

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
                 float        _eta0_try_sample_rate,
                 _COMP_T      _eta0_try_1st,
                 _COMP_T      _eta0_try_factor,
                 bool         _show_obj_each_iter):
         d(_dimension),
         v(_verbosity),
         eta0(_eta0),
         n_iter(_n_iter),
         err(_err),
         eta0_try_sample_rate(_eta0_try_sample_rate),
         eta0_try_1st(_eta0_try_1st),
         eta0_try_factor(_eta0_try_factor),
         show_obj_each_iter(_show_obj_each_iter),
         out_training_proc(NULL) {
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
   float        _eta0_try_sample_rate,
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
                           _eta0_try_sample_rate,
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
train(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y, _N_DAT_T* _x, _N_DAT_T _s_x) {
    training_data(_data, _n, _y, _x, _s_x);
    return train();
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train() {
    _DAT_DIM_T dim       = gd_param->dimension();
    char       verbosity = gd_param->verbosity();
    _COMP_T    eta0      = gd_param->init_learning_rate();
    if (!dim) {
        std::cerr << "WARNING: GD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    unsigned int n_iter = gd_param->num_of_iterations();
    _COMP_T err = gd_param->accuracy();
    bool show_obj_each_iter = gd_param->show_obj_each_iteration();
    _N_DAT_T n_sample = stat.num_of_samples();
    _N_DAT_T* x = NULL;
    if (n_sample < n) {
        _N_DAT_T* x = new _N_DAT_T[n_sample];
        stat.index_of_samples(x);
    }
    if (verbosity >= 1) {
        std::cout << "GD Training: \n\tData: " << n_sample << " samples, "
                  << dim << " feature dimensions.\n\tStopping Criterion: "
                  << n_iter << " iterations or accuracy higher than "
                  << err <<  "." << std::endl;
    }
    if (!eta0) try_learning_rate();
    if (verbosity == 1) {
        std::cout << "Training ... " << std::flush;
    }
    _COMP_T obj0 = std::numeric_limits<_COMP_T>::max();
    _COMP_T obj1;
    for (t = 0; t < n_iter; ++t) {
        if (verbosity >= 2) {
            std::cout << "Training: Iteration " << t << " ... " << std::flush;
        }
        compute_learning_rate(eta0);
        train_iteration(x, n_sample);
        obj1 = compute_obj();
        if (verbosity >= 2) {
            std::cout << "Done. ";
            if (show_obj_each_iter) {
                std::cout << "Objective = " << obj1 << ".";
            }
            std::cout << std::endl;
        }
        // DEBUG
        std::ostream* out = gd_param->ostream_of_training_process();
        if (out) {
            *out << obj1;
            *out << " ";
            ostream_param(*out);
            *out << "\n";
        }
        if (obj0 - obj1 < err) break;
        else obj0 = obj1;
    }
    if (verbosity == 1) {
        std::cout << "Done." << std::endl;
    }
    if (verbosity >= 1) {
        if (t < n_iter) std::cout << "Training stopped with convergence.";
        else std::cout << "Max number of iterations has been reached.";
        std::cout << "\nGD Training: finished." << std::endl;
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
    if (verbosity >= 1)
        std::cout << "GD Testing: \n\tData: " << _n << " samples."
                  << std::endl;
    if (verbosity == 1) std::cout << "Testing ... " << std::flush;
    if (_x && _s_x) {
        for (_N_DAT_T i = 0; i < _s_x; ++i) {
            _COMP_T* dat_i = _data + dim * _x[i];
            if (verbosity >= 2)
                std::cout << "Testing: Sample " << i << " ... " << std::flush;
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
                std::cout << "Testing: Sample " << i << " ... " << std::flush;
            _y[i] = test_one(dat_i, dim);
            if (verbosity >= 2)
                std::cout << "Done. Predicted label: " << _y[i] << "."
                          << std::endl;
        }
    }
    if (verbosity == 1) std::cout << "Done." << std::endl;
    if (verbosity >= 1) std::cout << "GD Testing: finished." << std::endl;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
try_learning_rate() {
    if (!gd_param->dimension()) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }

    _COMP_T eta0_try1, eta0_try2, obj_try1, obj_try2;
    char verbosity = gd_param->verbosity();
    if (verbosity >= 1) {
        std::cout << "No initial learning rate specified. Deciding automatically... ";
        if (verbosity >= 2) std::cout << std::endl;
        else std::cout.flush();
    }

    _SUPV_T  s_labelset  = stat.num_of_labels();
    _N_DAT_T n_subsample = (_N_DAT_T)(gd_param->learning_rate_sample_rate()
                            * stat.num_of_samples());
    _N_DAT_T* sub_x_i = new _N_DAT_T[n_subsample];
    stat.rand_index(sub_x_i, n_subsample);

    LabelStat<_SUPV_T, _N_DAT_T> stat_new(y, n_subsample, sub_x_i);

    eta0_try1 = gd_param->learning_rate_1st_try();
    if (verbosity >= 2)
        std::cout << "\tTrying eta0 = " << eta0_try1 << "... " << std::flush;
    GD* tempGD  = get_temp_dup();
    tempGD->eta = eta0_try1;
    tempGD->stat = stat_new;
    tempGD->train_iteration(sub_x_i, n_subsample);
    obj_try1 = tempGD->compute_obj();
    delete tempGD;
    if (verbosity >= 2) std::cout << "Done. Obj = " << obj_try1 << std::endl;
    _COMP_T eta0_try_factor = gd_param->learning_rate_try_factor();
    eta0_try2 = eta0_try1 * eta0_try_factor;
    if (verbosity >= 2)
        std::cout << "\tTrying eta0 = " << eta0_try2 << "... " << std::flush;
    tempGD = get_temp_dup();
    tempGD->eta = eta0_try2;
    tempGD->stat = stat_new;
    tempGD->train_iteration(sub_x_i, n_subsample);
    obj_try2 = tempGD->compute_obj();
    delete tempGD;
    if (verbosity >= 2)
        std::cout << "Done. Obj = " << obj_try2 << std::endl;
    if (obj_try1 < obj_try2) {
        eta0_try_factor = 1 / eta0_try_factor;
        obj_try2        = obj_try1;
        eta0_try2       = eta0_try1;
    }
    do {
        obj_try1  = obj_try2;
        eta0_try1 = eta0_try2;
        eta0_try2 = eta0_try1 * eta0_try_factor;
        if (verbosity >= 2)
            std::cout << "\tTrying eta0 = " << eta0_try2 << "... " << std::flush;
        tempGD = get_temp_dup();
        tempGD->eta = eta0_try2;
        tempGD->stat = stat_new;
        tempGD->train_iteration(sub_x_i, n_subsample);
        obj_try2 = tempGD->compute_obj();
        delete tempGD;
        if (verbosity >= 2)
            std::cout << "Done. Obj = " << obj_try2 << std::endl;
    } while (obj_try1 > obj_try2);
    gd_param->init_learning_rate(eta0_try1);
    if (verbosity >= 2)
        std::cout << "Setting eta0 = " << eta0_try1 << std::endl;
    else if (verbosity >= 1)
        std::cout << "Done. eta0 = " << eta0_try1 << std::endl;

    delete[] sub_x_i;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGDParam::SGDParam(unsigned int _n_epoch,
                   bool         _show_obj_each_epoch):
          n_epoch(_n_epoch),
          show_obj_each_epoch_(_show_obj_each_epoch) {
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGD(_DAT_DIM_T   _dimension,
    char         _verbosity,
    _COMP_T      _eta0,
    unsigned int _n_epoch,
    unsigned int _n_iter,
    _COMP_T      _err,
    float        _eta0_try_sample_rate,
    _COMP_T      _eta0_try_1st,
    _COMP_T      _eta0_try_factor,
    bool         _show_obj_each_epoch,
    bool         _show_obj_each_iter)
   :GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>(_dimension,
                                               _verbosity,
                                               _eta0,
                                               _n_iter,
                                               _err,
                                               _eta0_try_sample_rate,
                                               _eta0_try_1st,
                                               _eta0_try_factor,
                                               _show_obj_each_iter),
    alloc_sgd_param(true) {
    sgd_param = new SGDParam(_n_epoch, _show_obj_each_epoch);
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGD(typename GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::GDParam& _gd_param, SGDParam& _sgd_param)
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
    _DAT_DIM_T   dim         = this->gd_param->dimension();
    char         verbosity   = this->gd_param->verbosity();
    _COMP_T      err         = this->gd_param->accuracy();
    unsigned int n_epoch     = sgd_param->num_of_epoches();
    bool show_obj_each_epoch = sgd_param->show_obj_each_epoch();
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    _N_DAT_T     n_sample = this->stat.num_of_samples();
    if (verbosity >= 1) {
        std::cout << "SGD Training: \n\tData: " << n_sample << " samples, "
                  << dim << " feature dimensions.\n\tStopping Criterion: "
                  << n_epoch << " epoches or accuracy higher than "
                  << err <<  "." << std::endl;
    }
    if (!this->gd_param->init_learning_rate()) this->try_learning_rate();
    this->t = 0;
    _N_DAT_T* x = new _N_DAT_T[n_sample];
    _COMP_T obj0 = std::numeric_limits<_COMP_T>::max();
    _COMP_T obj1;
    while (this->t < n_epoch * n_sample) {
        if (verbosity >= 2)
            std::cout << "Shuffling the data set... " << std::flush;
        // re-shuffle the data
        this->stat.rand_index(x, n_sample);
        if (verbosity >= 2) std::cout << "Done." << std::endl; 
        if (verbosity >= 1) {
            std::cout << "Training: Epoch = " << this->t / n_sample + 1  << " ... ";
            if (verbosity >= 3) std::cout << std::endl;
            else std::cout.flush();
        }
        train_epoch(x, n_sample);
        obj1 = this->compute_obj();
        if (verbosity >= 1 && verbosity < 3) {
            std::cout << "Done.";
            if (show_obj_each_epoch) {
                std::cout << " Objective = " << obj1;
            }
            std::cout << std::endl;
        }
        if (obj0 - obj1 < err) break;
        else obj0 = obj1;
    }
    if (verbosity >= 1) {
        std::cout << "SGD Training: finished." << std::endl;
    }
    delete[] x;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train_epoch(_N_DAT_T* x, _N_DAT_T s_x) {
    _DAT_DIM_T dim          = this->gd_param->dimension();
    char       verbosity    = this->gd_param->verbosity();
    _COMP_T    eta0         = this->gd_param->init_learning_rate();
    bool show_obj_each_iter = sgd_param->show_obj_each_iteration();
    for (_N_DAT_T i = 0; i < s_x; ++i, ++this->t) {
        _N_DAT_T ind = x[i];
        if (verbosity >= 3) {
            std::cout << "\tSGD training through sample " << ind
                      << " (" << i+1 << "/" << s_x << ")... " << std::flush;
        }
        compute_learning_rate(eta0);
        train_one(this->data + dim * ind, dim);
        _COMP_T obj = 0;
        if (verbosity >= 3) {
            std::cout << "Done.";
            if (show_obj_each_iter) {
                obj = this->compute_obj();
                std::cout << " Objective = " << obj;
            }
            std::cout << std::endl;
        }

        // DEBUG
        std::ostream* out = this->gd_param->ostream_of_training_process();
        if (out) {
            if (obj) *out << obj;
            else *out << this->compute_obj();
            *out << " ";
            this->ostream_param(*out);
            *out << "\n";
        }
    }
    return *this;
}

# endif
