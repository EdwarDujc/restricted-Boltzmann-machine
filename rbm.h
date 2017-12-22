#include <iostream>

using namespace std;

class RBM
{
    public:
    size_t N;
    size_t n_visible;
    size_t n_hidden;
    double **W;
    double *hbias;
    double *vbias;

    RBM(size_t,size_t,size_t,double**,double*,double*);
    ~RBM();

    void contrastive_divergence(int *,double,int);
    void sample_h_given_v(int*,double*,int*);
    double sigmoid(double);
    double Vtoh_sigm(int* ,double* ,int );
    void gibbs_hvh(int* ,double* ,int* ,double* ,int* );
    double HtoV_sigm(int* ,int ,int );
    void sample_v_given_h(int* ,double* ,int* );
    void reconstruct(int* ,double*);

    private:
};