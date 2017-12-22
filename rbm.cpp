#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cstring>
#include "RBM.h"

using namespace std;

void test_rbm();
double uniform(double ,double);
double binomial(double );

int main()
{
    test_rbm();
	system("pause");
    return 0;
}

//start
void test_rbm()
{
    srand(0);
    size_t train_N = 6;
    size_t test_N = 2;
    size_t n_visible = 6;
    size_t n_hidden = 3;

    double learning_rate = 0.1;
    int training_num = 1000;
    int k = 1;

    int train_data[6][6] = {
    {1, 1, 1, 0, 0, 0},
    {1, 0, 1, 0, 0, 0},
    {1, 1, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 0},
    {0, 0, 1, 0, 1, 0},
    {0, 0, 1, 1, 1, 0}
    };

    RBM rbm(train_N,n_visible,n_hidden,NULL,NULL,NULL);    //configure

    for(size_t j=0;j<training_num;j++)
    {
        for(size_t i=0;i<train_N;i++)
        {
            rbm.contrastive_divergence(train_data[i],learning_rate,k);  //CDK algorithm training set
        }
    }

    //test set
    int test_data[2][6] = {
        {1,1,0,0,0,0},
        {0,0,0,1,1,0}
    };

    double reconstructed_data[2][6];

    for(size_t i=0;i<test_N;i++)
    {
        rbm.reconstruct(test_data[i],reconstructed_data[i]);    
        for(size_t j=0;j<n_visible;j++)
        {
            cout << reconstructed_data[i][j] << "  ";
        }
        cout << endl;
    }
}

//reconstruct data
void RBM::reconstruct(int* test_data,double* reconstructed_data)
{
    double* h = new double[n_hidden];
    double temp;

    for(size_t i=0;i<n_hidden;i++)
    {
        h[i] = Vtoh_sigm(test_data,W[i],hbias[i]);
    }

    for(size_t i=0;i<n_visible;i++)
    {
        temp = 0.0;
        for(size_t j=0;j<n_hidden;j++)
        {
            temp += W[j][i] * h[j];
        }
        temp += vbias[i];
        reconstructed_data[i] =  sigmoid(temp);
    }
    delete[] h;
}


void RBM::contrastive_divergence(int *train_data,double learning_rate,int k)
{
    double* ph_sigm_out = new double[n_hidden];
    int* ph_sample = new int[n_hidden];
    double* nv_sigm_outs = new double[n_visible];
    int* nv_samples = new int[n_visible];
    double* nh_sigm_outs = new double[n_hidden];
    int* nh_samples = new int[n_hidden];

    sample_h_given_v(train_data,ph_sigm_out,ph_sample);        //get h0

    for(size_t i=0;i<k;i++)
    {
        if(i == 0)
        {
            gibbs_hvh(ph_sample,nv_sigm_outs,nv_samples,nh_sigm_outs,nh_samples); //get V1,h1
        }
        else
        {
            gibbs_hvh(nh_samples,nv_sigm_outs,nv_samples,nh_sigm_outs,nh_samples);
        }
    }

    //update weights, biases, v0 represents original data x
    //h0 is ph_sigm_out£¬h0=p(h|v0)
    //v1=p(v|h0)
    //h1 similar as above
    for(size_t i=0;i<n_hidden;i++)
    {
        for(size_t j=0;j<n_visible;j++)
        {
           
            W[i][j] += learning_rate * (ph_sigm_out[i] * train_data[j] - nh_sigm_outs[i] * nv_samples[j]) / N;
        }
        hbias[i] += learning_rate * (ph_sample[i] - nh_sigm_outs[i]) / N;
    }

    for(size_t i=0;i<n_visible;i++)
    {
        vbias[i] += learning_rate * (train_data[i] - nv_samples[i]) / N;
    }

    delete[] ph_sigm_out;
    delete[] ph_sample;
    delete[] nv_sigm_outs;
    delete[] nv_samples;
    delete[] nh_sigm_outs;
    delete[] nh_samples;
}

void RBM::gibbs_hvh(int* ph_sample,double* nv_sigm_outs,int* nv_samples,double* nh_sigm_outs,int* nh_samples)
{
    sample_v_given_h(ph_sample,nv_sigm_outs,nv_samples);
    sample_h_given_v(nv_samples,nh_sigm_outs,nh_samples);
}

void RBM::sample_v_given_h(int* h0_sample,double* nv_sigm_outs,int* nv_samples)
{
    for(size_t i=0;i<n_visible;i++)
    {
        nv_sigm_outs[i] =  HtoV_sigm(h0_sample,i,vbias[i]);
        nv_samples[i] = binomial(nv_sigm_outs[i]);
    }
}

double RBM::HtoV_sigm(int* h0_sample,int i,int vbias)
{
    double temp = 0.0;
    for(size_t j=0;j<n_hidden;j++)
    {
        temp += W[j][i] * h0_sample[j];
    }
    temp += vbias;
    return sigmoid(temp);
}

void RBM::sample_h_given_v(int* train_data,double* ph_sigm_out,int* ph_sample)
{
    for(size_t i=0;i<n_hidden;i++)
    {
        ph_sigm_out[i] = Vtoh_sigm(train_data,W[i],hbias[i]);
        ph_sample[i] = binomial(ph_sigm_out[i]);
    }
}

double binomial(double p)
{
    if(p<0 || p>1){
        return 0;
    }
    double r = rand()/(RAND_MAX + 1.0);
    if(r < p)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

double RBM::Vtoh_sigm(int* train_data,double* W,int hbias)
{
    double temp = 0.0;
    for(size_t i=0;i<n_visible;i++)
    {
        temp += W[i] * train_data[i];
    }
    temp += hbias;
    return sigmoid(temp);
}

double RBM::sigmoid(double x)
{
    return 1.0/(1.0 + exp(-x));
}

RBM::RBM(size_t train_N,size_t n_v,size_t n_h,double **w,double *hb,double *vb)
{
    N = train_N;
    n_visible = n_v;
    n_hidden = n_h;

    if(w == NULL)
    {
        W = new double*[n_hidden];
        double a = 1.0/n_visible;
        for(size_t i=0;i<n_hidden;i++)
        {
            W[i] = new double[n_visible];
        }
        for(size_t i=0;i<n_hidden;i++)
        {
            for(size_t j=0;j<n_visible;j++)
            {
                W[i][j] = uniform(-a,a);
            }
        }
    }
    else
    {
        W = w;
    }

    if(hb == NULL)
    {
        hbias = new double[n_hidden];
        for(size_t i=0;i<n_hidden;i++)
        {
            hbias[i] = 0.0;
        }
    }
    else
    {
        hbias = hb;
    }

    if(vb == NULL)
    {
        vbias = new double[n_visible];
        for(size_t i=0;i<n_visible;i++)
        {
            vbias[i] = 0.0;
        }
    }
    else
    {
        vbias = vb;
    }
}

RBM::~RBM()
{
    for(size_t i=0;i<n_hidden;i++)
    {
        delete[] W[i];
    }
    delete[] W;
    delete[] hbias;
    delete[] vbias;
}

double uniform(double min,double max)
{
    return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}