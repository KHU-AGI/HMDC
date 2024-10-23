#include "kmeans.h"

/*
C++ class implementation
*/

template <typename T>
inline torch::Tensor& 
KMeans::fit_predict(T&& X){
    this->fit(std::forward<T>(X)); return this->predict(std::forward<T>(X));}

template <typename T>
inline torch::Tensor&  
KMeans::predict(T&& X){
    this->update_labels(std::forward<T>(X)); return *this->labels;}

template <typename T>
inline void
KMeans::fit(T&& X){
    torch::NoGradGuard no_grad;
    this->init_centers(std::forward<T>(X));
    torch::Tensor old_centroids = torch::empty({this->n_clusters, X.size(1)}, X.options());
    for (int i = 0; i < this->max_iter; i++) {
        old_centroids.copy_(*this->centroids);
        this->update_labels(std::forward<T>(X)); this->update_centers(std::forward<T>(X));
        if (old_centroids.equal(*this->centroids)) break;}
        if (this->centroids->size(0) != this->n_clusters) {throw std::invalid_argument("!!! The number of clusters is not equal to n_clusters !!!");}}

template <typename T>
inline void
KMeans::init_centers(T&& X){
    try {
        this->centroids = std::make_unique<torch::Tensor>(torch::empty({this->n_clusters, X.size(1)}, X.options()));
        this->labels    = std::make_unique<torch::Tensor>(torch::empty({X.size(0)}, X.options().dtype(torch::kInt64)));
        this->distances = std::make_unique<torch::Tensor>(torch::empty({X.size(0), this->n_clusters}, X.options()));
        this->mask      = std::make_unique<torch::Tensor>(torch::empty({X.size(0)}, X.options().dtype(torch::kBool)));

        if (this->init_method == Random) *this->centroids = std::forward<T>(X)[torch::randperm(X.size(0)).slice(0, 0, this->n_clusters)];
        else if (this->init_method == kMeansPP) {
            this->centroids->index({0}) = X.index({torch::randint(X.size(0), {1}).template item<int>()}); // Choose the first centroid randomly
            compute_distance_matrix_out(this->distances->slice(1,0,1,1), std::forward<T>(X), this->centroids->slice(0,0,1,1)); // Compute the distance matrix
            auto min_dist = torch::zeros({X.size(0)}, X.options()) + 1e-8;
            auto min_idx = torch::empty({X.size(0)}, X.options().dtype(torch::kInt64));
            for (int i = 0; i < (this->n_clusters - 1); i++) {
                compute_distance_matrix_out(this->distances->slice(1,i,i+1,1), std::forward<T>(X), this->centroids->slice(0,i,i+1,1)); // Compute the distance matrix
                torch::min_out(min_dist, min_idx, this->distances->slice(1,0,i+1,1), 1, false); // Get the minimum distance to the nearest centroid
                this->centroids->index({i+1}) = X.index({(min_dist/min_dist.sum()).multinomial(1).template item<int>()}); // Choose the next centroid
            }}
        else {throw std::invalid_argument("Invalid init method");}}
    catch (std::exception &e) {printf("Error: %s\n", e.what());}}

template <typename T>
inline void
KMeans::update_centers(T&& X){
    torch::NoGradGuard no_grad;
    for (int i = 0; i < this->n_clusters; i++) {
        *this->mask = *this->labels == i;
        if (this->mask->sum().item<int>() == 0) continue;
        torch::mean_out(this->centroids->index({i}), std::forward<T>(X).index({*this->mask}), {0}, false);
        // this->centroids->index({i}) = X.index({mask}).mean({0}, false);
    }}

template <typename T>
inline void
KMeans::update_labels(T&& X){
    torch::NoGradGuard no_grad;
    // *this->labels = this->compute_distance_matrix(X, *this->centroids).argmin(1);}
    torch::argmin_out(*this->labels, this->compute_distance_matrix(std::forward<T>(X), *this->centroids), 1);}

template <typename T>
inline torch::Tensor& 
KMeans::compute_distance_matrix(T&& X){
    torch::NoGradGuard no_grad;
    for (int64_t i = 0; i <std::forward<T>(X).size(0); i += this->batch_size) {
        this->pairwise_distance_out(this->distances->slice(0, i, i+this->batch_size, 1), std::forward<T>(X).slice(0, i, i+this->batch_size, 1), *this->centroids);
        }return *this->distances;}

template <typename T1, typename T2>
inline torch::Tensor&
KMeans::compute_distance_matrix(T1&& X, T2&& Y){
    torch::NoGradGuard no_grad;
    for (int64_t i = 0; i < X.size(0); i += this->batch_size) {
        this->pairwise_distance_out(this->distances->slice(0, i, i+this->batch_size, 1), X.slice(0, i, i+this->batch_size, 1), Y);}
    return *this->distances;}

template <typename T1, typename T2, typename T3>
inline torch::Tensor&
KMeans::compute_distance_matrix_out(T1&& Out, T2&& X, T3&& Y){
    torch::NoGradGuard no_grad;
    auto _x_size = std::forward<T2>(X).size(0);
    for (int64_t i = 0; i < _x_size; i += this->batch_size) {
        this->pairwise_distance_out(Out.slice(0, i, i+this->batch_size, 1),
            std::forward<T2>(X).slice(0, i, i+this->batch_size, 1), std::forward<T3>(Y));
        }
    return Out;
}

template <typename T1, typename T2>
inline torch::Tensor&
KMeans::pairwise_distance(T1&& X, T2&& Y){
    if (this->dist_method == Euclidean) return torch::norm(std::forward<T1>(X).unsqueeze(1) - std::forward<T2>(Y).unsqueeze(0), 2, 2);
    else if (this->dist_method == Cosine) {
        auto norms_X = torch::norm(std::forward<T1>(X), 2, 1);
        auto norms_Y = torch::norm(std::forward<T2>(Y), 2, 1);
        auto cosine_sim = 1 - torch::clamp(torch::mm(std::forward<T1>(X), std::forward<T2>(Y).t()) / (norms_X * norms_Y.t()), -1.0, 1.0);
        return cosine_sim;}
    else if (this->dist_method == Manhattan) return torch::norm(std::forward<T1>(X).unsqueeze(1) - std::forward<T2>(Y), 1, 2);
    else throw std::invalid_argument("Invalid distance method");
}

template <typename T1, typename T2, typename T3>
inline torch::Tensor&
KMeans::pairwise_distance_out(T1&& Out, T2&& X, T3&& Y){
    if (this->dist_method == Euclidean) torch::norm_out(Out, std::forward<T2>(X).unsqueeze(1) - std::forward<T3>(Y).unsqueeze(0), 2, 2);
    else if (this->dist_method == Cosine) {
        auto norms_X = torch::norm(std::forward<T2>(X), 2, 1);
        auto norms_Y = torch::norm(std::forward<T3>(Y), 2, 1);
        auto cosine_sim = 1 - torch::clamp(torch::mm(std::forward<T2>(X), std::forward<T3>(Y).t()) / (norms_X.unsqueeze(1) * norms_Y.unsqueeze(0)), -1.0, 1.0);
        Out.copy_(cosine_sim);}
    else if (this->dist_method == Manhattan) torch::norm_out(Out, std::forward<T2>(X).unsqueeze(1) - std::forward<T3>(Y), 1, 2);
    else throw std::invalid_argument("Invalid distance method");
    return Out;
}

#ifdef __cplusplus
extern "C" {
#endif

PyObject * Kmeans_alloc(PyTypeObject *type, Py_ssize_t nitems)
    {auto ret = PyObject_New(KMeans, type); if (ret == NULL) return NULL; return (PyObject*)ret;}

void Kmeans_dealloc(KMeans *self){
    self->centroids.reset();
    self->labels.reset();
    self->distances.reset();
    self->mask.reset();}
    
PyObject * Kmeans_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
    {auto self = (KMeans*)Kmeans_alloc(type, 0); return (PyObject*)self;}

int Kmeans_init(KMeans *self, PyObject *args, PyObject *kwds){
    // Initialize the pointers to nullptr. Because the Python interpreter allocate memory WITHOUT calling the constructor.
    if (self->centroids != nullptr) *(long*)&(self->centroids)=0;
    if (self->labels != nullptr) *(long*)&(self->labels)=0;
    if (self->distances != nullptr) *(long*)&(self->distances)=0;
    if (self->mask != nullptr) *(long*)&(self->mask)=0;

    PyObject *n_clusters=nullptr, *max_iter=nullptr, *batchsize=nullptr, *random_state=nullptr;
    char *init_method_str=nullptr, *dist_method_str=nullptr;
    const char* kwlist[] = {"n_clusters", "max_iter", "batchsize", "mode", "init", "seed", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOssO", (char**)kwlist, &n_clusters, &max_iter, &batchsize, &dist_method_str, &init_method_str, &random_state)) return -1;

    self->n_clusters = Py_IsNone(n_clusters) ? throw std::invalid_argument("n_clusters is required") : PyLong_AsLong(n_clusters);
    self->max_iter = Py_IsNone(max_iter) ? throw std::invalid_argument("max_iter is required") : PyLong_AsLong(max_iter);
    self->batch_size = Py_IsNone(batchsize) ? throw std::invalid_argument("batchsize is required") : PyLong_AsLong(batchsize);

    std::srand(std::time(0));
    self->random_state = Py_IsNone(random_state) ? std::rand() : PyLong_AsLong(random_state);

    if (!strcmp(init_method_str, "kmeans++")) self->init_method = kMeansPP;
    else if (!strcmp(init_method_str, "random")) self->init_method = Random;
    else {printf("Unknown init_method_str: %s\n", init_method_str); return -1;}

    if (strcmp(dist_method_str, "euclidean") == 0) self->dist_method = Euclidean;
    else if (strcmp(dist_method_str, "cosine") == 0) self->dist_method = Cosine;
    else if (strcmp(dist_method_str, "manhattan") == 0) self->dist_method = Manhattan;
    else {printf("Unknown dist_method_str: %s\n", dist_method_str); return -1;}
    return 0;}

PyObject * Kmeans_repr(KMeans *self) {
    return PyUnicode_FromFormat("KMeans(n_clusters=%d, max_iter=%d, batch_size=%d, random_state=%d, init_method=%s, dist_method=%s)", 
        self->n_clusters, self->max_iter, self->batch_size, self->random_state, self->init_method, self->dist_method);}

PyObject *
fit(KMeans *self, PyObject *args) {
    PyObject *X;
    if (!PyArg_ParseTuple(args, "O:fit", &X)) return NULL;
    self->fit(THPVariable_Unpack(X));
    return Py_None;}

PyObject *
fit_predict(KMeans *self, PyObject *args) {
    PyObject *X;
    if (!PyArg_ParseTuple(args, "O:fit_predict", &X)) return NULL;
    return THPVariable_Wrap(self->fit_predict(THPVariable_Unpack(X)));}

PyObject *
predict(KMeans *self, PyObject *args) {
    PyObject *X;
    if (!PyArg_ParseTuple(args, "O:predict", &X)) return NULL;
    return THPVariable_Wrap(self->predict(THPVariable_Unpack(X)));}

PyObject *
init_centers(KMeans *self, PyObject *args) {
    PyObject *X;
    if (!PyArg_ParseTuple(args, "O:init_centers", &X)) return NULL;
    self->init_centers(THPVariable_Unpack(X));
    return THPVariable_Wrap(*self->centroids);}

PyObject *
update_centers(KMeans *self, PyObject *args) {
    PyObject *X;
    if (!PyArg_ParseTuple(args, "O:update_centers", &X)) return NULL;
    self->update_centers(THPVariable_Unpack(X));
    return THPVariable_Wrap(*self->centroids);}

PyObject *
update_labels(KMeans *self, PyObject *args) {
    PyObject *X;
    if (!PyArg_ParseTuple(args, "O:update_labels", &X)) return NULL;
    self->update_labels(THPVariable_Unpack(X));
    return Py_None;}

PyObject *
compute_distance_matrix(KMeans *self, PyObject *args){
    PyObject *X, *Y;
    if (!PyArg_ParseTuple(args, "OO:compute_distance_matrix", &X, &Y)) return NULL;
    return THPVariable_Wrap(self->compute_distance_matrix(THPVariable_Unpack(X), THPVariable_Unpack(Y)));}

PyObject *
get_centroids(KMeans *self, PyObject *args) {
    return THPVariable_Wrap(self->centroids->clone());}

#ifdef __cplusplus
}
#endif

PyMODINIT_FUNC
PyInit_kmeans(void)
{
    auto mod = PyModule_Create(&KMeansModule);
    PyModule_AddType(mod, &KMeansType);
    if (mod == NULL) return NULL;
    return mod;
}