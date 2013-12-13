# ifndef _SGD_H
# define _SGD_H

# include <iostream>
# include <cstdlib>
# include <cstring>
# include <ctime>
// # include <forward_list>

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
    /// @param[in] dat The matrix that holds all the training data.
    /// @param[in] n   Number of samples in the matrix.
    /// @param[in] y   The array containing labels of each sample.
    /// @param[in] x   If only a subset of data is used as training data, "x" is 
    ///                an index array of length "s_x" that holds all the indexes
    ///                of training samples.
    /// @param[in] s_x Number of training samples.
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    training_data(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y,
                  _N_DAT_T* _x = NULL, _N_DAT_T _s_x = 0);
    /// Train the parameters with ALL the "n" input data of dimension "dim" and 
    /// the corresponding supervising data "y". 
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
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
    _COMP_T*  data;       ///< data buffer
    _N_DAT_T  n;          ///< total number of samples in the buffer
    _SUPV_T*  y;          ///< labels
    _N_DAT_T* x;          ///< indexes of training data
    _N_DAT_T  s_x;        ///< number of training samples

    GDParam*  gd_param;
    // temporary parameters during learning
    _N_DAT_T t;            ///< current pass (number of data samples passed)
    _COMP_T  eta;          ///< current learning rate

    /// Determine the initial learning rate eta0 automatically according to the 
    /// given data set.
    GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    try_learning_rate();
    virtual _COMP_T compute_learning_rate() = 0;
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>&
    train_iteration() = 0;

    /// Test one data sample.
    virtual _SUPV_T test_one(_COMP_T* dat_i) const = 0;

    /// Randomly choose "m" numbers out of an array "index" of length "n" and 
    /// put them at the beginning of the array.
    virtual GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    rand_data_index(_N_DAT_T* index, _N_DAT_T m, _N_DAT_T n);

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
    SGD(GDParam& _gd_param, SGDParam& _sgd_param);
    SGD(SGD& some);
    virtual ~SGD();

    /// Train the parameters with ALL the "n" input data of dimension "dim" and 
    /// the corresponding supervising data "y". 
    /// 
    /// The order of the data samples are automatically shuffled before each 
    /// epoch. Every parameter is applied with update rule once for each input 
    /// data sample.
    SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train(_COMP_T* dat, _N_DAT_T n, _SUPV_T* y);

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
    train_epoch(_COMP_T* dat, _N_DAT_T n,
                _N_DAT_T* dat_idx, _N_DAT_T m, _SUPV_T* y);

    /// Update EVERY parameter once with one input data 
    /// (Sub-routine of SGD::train_epoch).
    virtual SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
    train_one(_COMP_T* dat, _N_DAT_T i, _N_DAT_T n, _SUPV_T* y) = 0;

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
   x(NULL),
   s_x(0),
   t(0),
   eta(_eta0),
   alloc_gd_param(true) {
    std::srand(std::time(NULL));
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
   x(NULL),
   s_x(0),
   gd_param(&_gd_param),
   t(0),
   alloc_gd_param(false) {
    std::srand(std::time(NULL));
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
   x(some.x),
   s_x(some.s_x),
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
    if (_s_x > _n) {
        std::cerr << "WARNING: GD: Invalid data set for training. "
                  << "Training data setting is skipped." << std::endl;
        return *this;
    }
    data = _data;
    n    = _n;
    y    = _y;
    x    = _x;
    s_x  = _s_x;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y, _N_DAT_T* _x, _N_DAT_T _s_x) {
    _DAT_DIM_T dim       = param->dimension();
    char       verbosity = param->verbosity();
    if (!dim) {
        std::cerr << "WARNING: GD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    training_data(_COMP_T* _data, _N_DAT_T _n, _SUPV_T* _y,
                  _N_DAT_T* _x, _N_DAT_T _s_x);
    unsigned int n_iter = gd_param->num_of_iterations();
    _COMP_T err = gd_param->accuracy();
    bool show_obj_each_iter = sgd_param->show_obj_each_iteration();
    if (verbosity >= 1) {
        std::cout << "GD Training: \n\tData: " << _n << " samples, "
                  << dim << " feature dimensions.\n\tStopping Criterion: "
                  << n_iter << " iterations or accuracy higher than "
                  << err <<  "." << std::endl;
    }
    if (!gd_param->init_learning_rate()) try_learning_rate();
    if (verbosity == 1) {
        std::cout << "Training ... " << std::flush;
    }
    if (verbosity >= 2) {
        std::cout << "Training: Iteration 0 ... " << std::flush;
    }
    train_iteration(_data, _n, y);
    _COMP_T obj0 = compute_obj(_data, _n, y);
    _COMP_T obj1;
    if (verbosity >= 2) {
        std::cout << "Done. ";
        if (show_obj_each_iter) {
            std::cout << "Objective = " << obj0 << ".";
        }
        std::cout << std::endl;
    }
    unsigned int i;
    for (i = 1; i < n_iter; ++i) {
        if (verbosity >= 2) {
            std::cout << "Training: Iteration " << i << " ... " << std::flush;
        }
        train_iteration(_data, _n, y);
        obj1 = compute_obj(_data, _n, y);
        if (verbosity >= 2) {
            std::cout << "Done. ";
            if (show_obj_each_iter) {
                std::cout << "Objective = " << obj0 << ".";
            }
            std::cout << std::endl;
        }
        if (obj0 - obj1 < err) break;
        else obj0 = obj1;
    }
    if (verbosity == 1) {
        std::cout << "Done." << std::endl;
    }
    if (verbosity >= 1) {
        if (i < n_iter) std::cout << "Training stopped at iteration " << i << ".";
        else std::cout << "Max number of iterations has been reached.";
        std::cout << "\nGD Training: finished." << std::endl;
    }
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
test(_COMP_T* dat, _N_DAT_T n, _SUPV_T* y) {
    _DAT_DIM_T dim       = gd_param->dimension();
    char       verbosity = gd_param->verbosity();    
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }
    _COMP_T* dat_i = dat;
    for (_N_DAT_T i = 0; i < n; ++i, dat_i += dim) {
        if (verbosity >= 1)
            std::cout << "GD Testing: \n\tData: " << n << " samples."
                      << std::endl;
        if (verbosity == 1) std::cout << "Testing ... " << std::flush;
        if (verbosity >= 2)
            std::cout << "Testing: Sample " << i << " ... " << std::flush;
        y[i] = test_one(dat_i);
        if (verbosity >= 2)
            std::cout << "Done. Predicted label: " << y[i] << "."
                      << std::endl;
        if (verbosity == 1) std::cout << "Done." << std::endl;
        if (verbosity >= 1) std::cout << "GD Testing: finished." << std::endl;
    }
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
try_learning_rate(_COMP_T* dat, _N_DAT_T n, _SUPV_T* y) {
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

    Histogram<_SUPV_T, _N_DAT_T> hist(y, n);
    _SUPV_T    s_labelset     = hist.length();
    _N_DAT_T** x_i_of_label   = new _N_DAT_T*[s_labelset];
    _N_DAT_T*  count_of_label = new _N_DAT_T[s_labelset];
    std::memset(count_of_label, 0, s_labelset * sizeof(_N_DAT_T));
    for (_SUPV_T i = 0; i < s_labelset; ++i) {
        x_i_of_label[i] = new _N_DAT_T[hist[i]];
    }
    for (_N_DAT_T i = 0; i < n; ++i) {
        x_i_of_label[y[i] - 1][count_of_label[y[i] - 1]++] = i;
    }
    float sample_rate = gd_param->learning_rate_sample_rate();
    _N_DAT_T  n_subsample = 0;
    for (_SUPV_T i = 0; i < s_labelset; ++i) {
        count_of_label[i] = (_N_DAT_T)(sample_rate / count_of_label[i]);
        n_subsample += count_of_label[i];
        rand_data_index(x_i_of_label[i], count_of_label[i], hist[i]);
    }
    _N_DAT_T* sub_x_i = new _N_DAT_T[n_subsample];
    n_subsample = 0;
    for (_SUPV_T i = 0; i < s_labelset; ++i) {
        for (_N_DAT_T j = 0; j < count_of_label[i]; ++j) {
            sub_x_i[n_subsample++] = x_i_of_label[i][j];
        }
    }

    for (_SUPV_T i = 0; i < s_labelset; ++i) {
        delete[] x_i_of_label[i];
    }
    delete[] x_i_of_label;
    delete[] count_of_label;

    eta0_try1 = gd_param->learning_rate_1st_try();
    if (verbosity >= 2)
        std::cout << "\tTrying eta0 = " << eta0_try1 << "... " << std::flush;
    GD* tempGD = get_temp_dup(); tempSGD->eta = eta0_try1;
    tempGD->train_iteration(dat, n, y, sub_x_i, n_subsample);
    obj_try1 = tempSGD->compute_obj(dat, n, y);
    delete tempSGD;
    if (verbosity >= 2)
        std::cout << "Done. Obj = " << obj_try1 << std::endl;
    _COMP_T eta0_try_factor = gd_param->learning_rate_try_factor();
    eta0_try2 = eta0_try1 * eta0_try_factor;
    if (verbosity >= 2)
        std::cout << "\tTrying eta0 = " << eta0_try2 << "... " << std::flush;
    tempGD = get_temp_dup(); tempSGD->eta = eta0_try2;
    tempGD->train_iteration(dat, n, y, sub_x_i, n_subsample);
    obj_try2 = tempSGD->compute_obj(dat, n, y);
    delete tempSGD;
    if (verbosity >= 2)
        std::cout << "Done. Obj = " << obj_try2 << std::endl;
    bool try_larger = obj_try1 > obj_try2;
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
        tempSGD = get_temp_dup(); tempSGD->eta = eta0_try2;
        tempSGD->train_iteration(dat, n, y, sub_x_i, n_subsample);
        obj_try2 = tempSGD->compute_obj(dat, n, y);
        delete tempSGD;
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
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
GD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
rand_data_index(_N_DAT_T* index, _N_DAT_T m, _N_DAT_T n) {
// WARNING: current implementation can generate random integers only less than 
// maximum value of type "int" because of the restrictions of rand() from C 
// standard library.
    for (_N_DAT_T i = 0; i < m; ++i) {
        _N_DAT_T temp_i = std::rand() % (n - i) + i;
        _N_DAT_T temp = index[temp_i];
        index[temp_i] = index[i];
        index[i] = temp;
    }
    // std::forward_list<_N_DAT_T> candidate;
    // for (_N_DAT_T i = 0; i < n; ++i) {
    //     candidate.push_front(i);
    // }
    // for (_N_DAT_T i = 0; i < n; ++i) {
    //     _N_DAT_T rand_i = std::rand() % ( n - i );
    //     typename std::forward_list<_N_DAT_T>::iterator it0 = 
    //         candidate.before_begin();
    //     typename std::forward_list<_N_DAT_T>::iterator it1 = 
    //         candidate.begin();
    //     for (_N_DAT_T i = 0; i < rand_i; ++i) {
    //         ++it0;
    //         ++it1;
    //     }
    //     index[i] = *it1;
    //     candidate.erase_after(it0);
    // }
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
SGDParam::SGDParam(unsigned int _n_epoch,
                   bool         _show_obj_each_epoch)
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
SGD(GDParam& _gd_param, SGDParam& _sgd_param)
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
train(_COMP_T* dat, _N_DAT_T n, _SUPV_T* y) {
    _DAT_DIM_T dim       = gd_param->dimension();
    char       verbosity = gd_param->verbosity();
    if (!dim) {
        std::cerr << "WARNING: SGD: Dimensionality has not been specified. "
                  << "Training process is skipped." << std::endl;
        return *this;
    }

    Histogram<_SUPV_T, _N_DAT_T> hist(y, n);

    _N_DAT_T* rand_dat_idx = new _N_DAT_T[n];   // buffer for random indexes
    unsigned int n_epoch = sgd_param->num_of_epoches();
    if (verbosity >= 1) {
        std::cout << "SGD Training: \n\tData: " << n << " samples, "
                  << dim << " feature dimensions.\n\tStopping Criterion: "
                  << n_epoch << " epoches." << std::endl;
    }
    if (!gd_param->init_learning_rate()) try_learning_rate(dat, n, y);
    bool show_obj_each_epoch = sgd_param->show_obj_each_epoch();
    while (t < n_epoch * n) {
        if (!(t % n)) {                       // during the start of each epoch, 
            if (verbosity >= 2) {
                std::cout << "Shuffling the data set... " << std::flush;
            }
            // re-shuffle the data
            rand_data_index(rand_dat_idx, y, n);
            if (verbosity >= 2) {
                std::cout << "Done." << std::endl; 
            }
            if (verbosity >= 1) {
                std::cout << "Training: Epoch = " << t / n + 1  << " ... ";
                if (verbosity >= 3) std::cout << std::endl;
                else std::cout.flush();
            }
        }
        train_epoch(dat, n, rand_dat_idx, n, y);
        if (verbosity >= 1 && verbosity < 3) {
            std::cout << "Done.";
            if (show_obj_each_epoch) {
                std::cout << " Objective = " << compute_obj(dat, n, y);
            }
            std::cout << std::endl;
        }
    }
    if (verbosity >= 1) {
        std::cout << "SGD Training: finished." << std::endl;
    }
    delete[] rand_dat_idx;
    return *this;
}

template <typename _COMP_T,
          typename _SUPV_T,
          typename _DAT_DIM_T,
          typename _N_DAT_T>
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>& 
SGD<_COMP_T, _SUPV_T, _DAT_DIM_T, _N_DAT_T>::
train_epoch(_COMP_T* dat, _N_DAT_T n,
            _N_DAT_T* dat_idx, _N_DAT_T m, _SUPV_T* y) {
    char verbosity = gd_param->verbosity();
    for (_N_DAT_T i = 0; i < m; ++i, ++t) {
        _N_DAT_T ind = dat_idx[i];
        if (verbosity >= 3) {
            std::cout << "\tSGD training through sample " << ind
                      << " (" << i+1 << "/" << m << ")... " << std::flush;
        }
        eta = compute_learning_rate();
        train_one(dat, ind, n, y);
        if (verbosity >= 3) {
            std::cout << "Done." << std::endl;
        }

        // DEBUG
        std::ostream* out = gd_param->ostream_of_training_process();
        if (out) {
            *out << compute_obj(dat, n, y) << " ";
            ostream_param(*out);
            *out << "\n";
        }
    }
    return *this;
}



# endif
