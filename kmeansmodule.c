#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define int_max 2147483647

double global_delta_uk = 1000000;
int num_vectors,size_vec;
double epsilon;

double* sumVector(double* p, double* q);
double** Kmeans(int k,int iter,double** vectors,double** centroids);
//double** read_file(char *input_filename);
double d(double* p, double* q);
double* divideVector(double* p, double num);

int isInt(const char* str);
void printVectors(double** res,int k);
double** update_centroids(double** centroids,int* vector_cluster,int k,double** vectors);
double** create_mat(int vec_num, int vec_size);
static PyObject* fit(PyObject *self, PyObject *args);


void printVectors(double** res,int k){
    int i;
    int j;
    for(i=0;i<k;i++){
        if(i != 0)
        {
            printf("\n");
        }
        for(j=0;j<size_vec;j++){
            printf("%.4f",res[i][j]);
            if(j != size_vec-1) {
                printf(",");
            }
        }
    }
}

int isInt(const char* str) {
    char* endptr;
    strtol(str, &endptr, 10);
    return (*str != '\0' && *endptr == '\0');
}

double d(double* p, double* q){
    double sum=0;
    int i;
    for(i=0;i<size_vec;i++){
        sum = sum + pow(p[i] - q[i],2);
    }
    return sqrt(sum);
}


double* divideVector(double* p, double num) {
    int i;
    double* div = malloc(size_vec * sizeof(double));
    assert(div);

    for (i = 0; i < size_vec; i++) {
        div[i] = p[i] / num;
    }

    return div;
}

double* sumVector(double* p, double* q){
    int i;
    double* sum = calloc(size_vec,sizeof(double));
    assert(sum);

    for (i = 0; i < size_vec; i++) {
        sum[i] = p[i] + q[i];
    }

    return sum;
}

double** update_centroids(double** centroids,int* vector_cluster,int k,double** vectors){
    int i;
    int j;
    int count;
    double maxdeltaUk;
    double deltauk;
    double** new_centroids = (double **)calloc(k,sizeof(double*));
    assert(new_centroids);

    for (i = 0 ; i < k ; i++)
    {
        new_centroids[i] = (double *)calloc(size_vec, sizeof(double));
        assert(new_centroids[i]);
    }
    maxdeltaUk = 0;
    deltauk = 0;
    for (i = 0; i < k; i++) {
        double* sum = (double *)calloc(size_vec, sizeof(double));
        assert(sum);
        count = 0;
        for (j = 0; j < num_vectors; j++) {
            if (vector_cluster[j] == i){
                sum = sumVector(vectors[j],sum);
                count++;
            }
        }
        new_centroids[i] = divideVector(sum,count);
        deltauk = d(new_centroids[i],centroids[i]);
        if (deltauk > maxdeltaUk){
            maxdeltaUk = deltauk;
        }
    }
    global_delta_uk = maxdeltaUk;
    return new_centroids;
}


double** Kmeans(int k,int iter,double** vectors,double** centroids){

    int j;
    int i;
    int l;
    i = 0;
    while (i < iter && global_delta_uk>epsilon){
        int* vector_cluster = (int*)calloc(num_vectors , sizeof(int));
        assert(vector_cluster);
        for (j = 0;j < num_vectors; j++) {
            double min = int_max;
            for (l = 0;l<k;l++){
                double distance = d(vectors[j],centroids[l]);
                if (distance < min){
                    min = distance;
                    vector_cluster[j] = l;
                }
            }
        }
        centroids = update_centroids(centroids,vector_cluster,k,vectors);
        i++;
        free(vector_cluster);

    }
    //free(vectors);
    return centroids;
}



double** parse_py_table_to_C(PyObject *lst, int vec_num, int vec_size)
{
    
    int row, col;
    double **data_points_c = create_mat(vec_num, vec_size);
    if (!data_points_c)
    {
        return NULL;
    }
    for (row = 0; row < vec_num ; row++)
    {
        PyObject *vector = PyList_GetItem(lst, row);
        for (col = 0 ; col < vec_size ; col++)
        {
            data_points_c[row][col] = PyFloat_AsDouble(PyList_GetItem(vector, col));
        }
    }
    return data_points_c;

}

double** create_mat(int vec_num, int vec_size){
    int i;
    double **mat;
    mat = calloc(vec_num, sizeof(double*));
    if (!mat)
    {
        return NULL;
    }
    for (i=0; i < vec_num; i++)
    {
        mat[i] = calloc(vec_size, sizeof(double));
        if (!mat[i])
        {
            return NULL;
        }
    }
    return mat;
}

PyObject* parse_centroids_to_py(double** centroids_c,int k)
{
    PyObject *centroid_py, *centroids_py;
    int i, j;
    centroids_py = PyList_New(0);
    /* ask if necessary*/
    if (!centroids_py)
    {
        return NULL;
    }
    for (i = 0; i < k ; i++)
    {
        centroid_py = PyList_New(0);
        if (centroid_py == NULL)
        {
            return NULL;
        }
        for (j = 0 ; j < size_vec ; j++)
        {
            PyList_Append(centroid_py, PyFloat_FromDouble(centroids_c[i][j]));
        }
        PyList_Append(centroids_py, centroid_py);
    }
    return centroids_py;
}

static PyObject* fit(PyObject *self, PyObject *args)
{
    
    int k, max_iter, total_vec_number, vec_size,i;
    double epsilon_p;
    double **data_points_c, **initial_centroids_c, **centroids_c;
    PyObject *data_points_py, *initial_centroids_py, *centroids_py;
    if (!PyArg_ParseTuple(args, "iidOOii", &k, &max_iter, &epsilon_p, &data_points_py, &initial_centroids_py, &total_vec_number, &vec_size))
    {
        return NULL;
    }
    size_vec = vec_size;
    data_points_c = parse_py_table_to_C(data_points_py, total_vec_number, size_vec);
    initial_centroids_c = parse_py_table_to_C(initial_centroids_py, k, size_vec);
    size_vec = vec_size;
    num_vectors = total_vec_number;
    epsilon = epsilon_p;
    
    centroids_c = Kmeans(k, max_iter, data_points_c, initial_centroids_c); // TODO: change
    centroids_py = parse_centroids_to_py(centroids_c,k);
    

    /*free vecs: */
    /*  free(data_points_c[0]); */
/* problem with this memory free loop? */
    for (i = 0; i < total_vec_number ; i++)
    {
        free(data_points_c[i]);
    }
    free(data_points_c);
    free(centroids_c);
    return centroids_py;
}


/* methods from class*/

static PyMethodDef capiMethods[] = {
        {"fit",
                (PyCFunction) fit,
                     METH_VARARGS,
                PyDoc_STR("returns calculated centroids for given matrix of data points")},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        NULL,
        -1,
        capiMethods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}