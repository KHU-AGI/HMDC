#include <ctime>
#include <memory>
#include <cstdlib>
#include <Python.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define KmeansObject_Check(v) Py_IS_TYPE(v, &KmeansType)

enum InitMethod { kMeansPP, Random };
enum DistMethod { Euclidean, Cosine, Manhattan };

namespace at{
    // Overload the functions to use reference of rvalues.
    inline at::Tensor & mean_out(at::Tensor && out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt) {
        return at::_ops::mean_out::call(self, dim, keepdim, dtype, out);
    }
    inline std::tuple<at::Tensor &,at::Tensor &> min_out(at::Tensor & out, at::Tensor & out_idx, at::Tensor && self, int64_t dim, bool keepdim=false) {
        return at::_ops::min_dim_min::call(self, dim, keepdim, out, out_idx);
    }
    inline at::Tensor norm(const at::Tensor && self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim=false) {
        return at::_ops::norm_ScalarOpt_dim::call(self, p, dim, keepdim);
    }
}

class KMeans {
public:    
    PyObject_HEAD;
    PyObject* x_attr; /* Attributes dictionary */

    int n_clusters;
    int max_iter;
    int batch_size;
    int random_state;
    int init_method;
    int dist_method;
    std::unique_ptr<torch::Tensor> centroids;
    std::unique_ptr<torch::Tensor> labels;
    std::unique_ptr<torch::Tensor> distances;
    std::unique_ptr<torch::Tensor> mask;

    KMeans() = default;
    KMeans(int n_clusters, int max_iter, int batchsize, int random_state, int init_method, int dist_method);
    
    template <typename T>
    inline torch::Tensor& fit_predict(T&& X);
    template <typename T>
    inline torch::Tensor& predict(T&& X);
    template <typename T>
    inline void fit(T&& X);
    template <typename T>
    inline void init_centers(T&& X);
    template <typename T>
    inline void update_centers(T&& X);
    template <typename T>
    inline void update_labels(T&& X);
    template <typename T>
    inline torch::Tensor& compute_distance_matrix(T&& X);
    template <typename T1, typename T2>
    inline torch::Tensor& compute_distance_matrix(T1&& X, T2&& Y);
    template <typename T1, typename T2, typename T3>
    inline torch::Tensor& compute_distance_matrix_out(T1&& Out, T2&& X, T3&& Y);
    template <typename T1, typename T2>
    inline torch::Tensor& pairwise_distance(T1&& X, T2&& Y);
    template <typename T1, typename T2, typename T3>
    inline torch::Tensor& pairwise_distance_out(T1&& Out, T2&& X, T3&& Y);
};

#ifdef __cplusplus
extern "C" {
#endif

PyObject *ErrorObject;
static PyObject * Kmeans_alloc(PyTypeObject *type, Py_ssize_t nitems=0);
static void Kmeans_dealloc(KMeans *self);
static int Kmeans_init(KMeans *self, PyObject *args, PyObject *kwds);
static PyObject * Kmeans_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
// static void Kmeans_finalize(KMeans *self);
static PyObject * fit(KMeans *self, PyObject *args);
static PyObject * fit_predict(KMeans *self, PyObject *args);
static PyObject * predict(KMeans *self, PyObject *args);
static PyObject * init_centers(KMeans *self, PyObject *args);
static PyObject * update_centers(KMeans *self, PyObject *args);
static PyObject * update_labels(KMeans *self, PyObject *args);
static PyObject * compute_distance_matrix(KMeans *self, PyObject *args);
static PyObject * get_centroids(KMeans *self, PyObject *args);

static PyMethodDef KMeans_methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, "Fits the model to the data."},
    {"fit_predict", (PyCFunction)fit_predict, METH_VARARGS, "Fits the model to the data and then predicts the clusters."},
    {"predict", (PyCFunction)predict, METH_VARARGS, "Predicts the clusters for the data."},
    {"init_centers", (PyCFunction)init_centers, METH_VARARGS, "Initializes the cluster centers."},
    {"update_centers", (PyCFunction)update_centers, METH_VARARGS, "Updates the cluster centers."},
    {"update_labels", (PyCFunction)update_labels, METH_VARARGS, "Updates the labels for the data points."},
    {"compute_distance_matrix", (PyCFunction)compute_distance_matrix, METH_VARARGS, "Computes the distance matrix."},
    {"get_centroids", (PyCFunction)get_centroids, METH_VARARGS, "Returns the cluster centers."},
    {NULL, NULL} /* sentinel */
};

PyTypeObject KMeansType = {
    /* The ob_type field must be initialized in the module init function
     * to be portable to Windows without using C++. */
    PyVarObject_HEAD_INIT(NULL, 0)
    "kmeans.KMeans",            /*tp_name*/
    sizeof(KMeans),             /*tp_basicsize*/
    0,                          /*tp_itemsize*/
    /* methods */
    (destructor)Kmeans_dealloc, /*tp_dealloc*/
    0,                          /*tp_vectorcall_offset*/
    0,                          /*tp_getattr*/
    0,                          /*tp_setattr*/
    0,                          /*tp_as_async*/
    0,                          /*tp_repr*/
    0,                          /*tp_as_number*/
    0,                          /*tp_as_sequence*/
    0,                          /*tp_as_mapping*/
    0,                          /*tp_hash*/
    0,                          /*tp_call*/
    0,                          /*tp_str*/
    0,                          /*tp_getattro*/
    0,                          /*tp_setattro*/
    0,                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,         /*tp_flags*/
    0,                          /*tp_doc*/
    0,                          /*tp_traverse*/
    0,                          /*tp_clear*/
    0,                          /*tp_richcompare*/
    0,                          /*tp_weaklistoffset*/
    0,                          /*tp_iter*/
    0,                          /*tp_iternext*/
    KMeans_methods,             /*tp_methods*/
    0,                          /*tp_members*/
    0,                          /*tp_getset*/
    0,                          /*tp_base*/
    0,                          /*tp_dict*/
    0,                          /*tp_descr_get*/
    0,                          /*tp_descr_set*/
    0,                          /*tp_dictoffset*/
    (initproc)Kmeans_init,      /*tp_init*/
    (allocfunc)Kmeans_alloc,    /*tp_alloc*/
    (newfunc)Kmeans_new,        /*tp_new*/
    0,                          /*tp_free*/
    0,                          /*tp_is_gc*/
};
/* --------------------------------------------------------------------- */

const char *module_doc = "This module provides an kmeans interface for Pytorch";

struct PyModuleDef KMeansModule = {
    PyModuleDef_HEAD_INIT,
    "kmeans",
    module_doc,
    0,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL};
#ifdef __cplusplus
}
#endif
PyMODINIT_FUNC PyInit_kmeans(void);