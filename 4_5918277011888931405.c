#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <windows.h>

typedef complex  float fcomplex;

#define W(N,k) (cexp(-2.0f * M_PI * I * (float)(k) / (float)(N))) /// To compute the twiddle factor
#define log2(a) ((int)(log(a)/log(2)))

 fcomplex **twiddle;    /// General two dimensions matrix


int main(){
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    double interval;
    int j;
    double c;
    double ci;
    fcomplex x[4] = {1,2,3,4};//,3,4,5,6,7,8};//,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16};
  //  for (j=1;j<=64;j++)
    //    x[j] = (float)j;
    fcomplex y[4] ;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    pre_fft(4);
    fft(x,y,0,1,4);
        QueryPerformanceCounter(&end);
    int i= 0;
    for(i=0;i<4;i++)
        {
           c =crealf(y[i]);
            int j = i;
            printf("%f ",c);
            ci =cimagf(y[j]);
            printf("+ %fi\n",ci);
        }
    interval = (double)(end.QuadPart - start.QuadPart)/frequency.QuadPart;
    printf("\n Time:%f\n",interval);
    return 0;
}

 void pre_fft(int N) {
    int i;

    int n_temp = log2(N)-2;
    twiddle = malloc(n_temp * sizeof(fcomplex *));  ///memory allocate
    for(i=0;i<n_temp;i++) {
        int n = N / pow(2,i);
        twiddle[i] = _mm_malloc(n/2 * sizeof(fcomplex), 16);

        int j;
        for(j=0;j<n/2;j+=4) {
            fcomplex w[4];
            int k;
            for(k=0;k<4;k++)
                w[k] = W(n,j+k);
            twiddle[i][j] = creal(w[0])+creal(w[1])*I;
            twiddle[i][j+1] = creal(w[2])+creal(w[3])*I;
            twiddle[i][j+2] = cimag(w[0])+cimag(w[1])*I;
            twiddle[i][j+3] = cimag(w[2])+cimag(w[3])*I;
            }
    }
 }

 void fft(fcomplex *in, fcomplex *out, int log2step, int step, int N) {
    if(N == 2) {
                    out[0] = in[0] + in[step];
                out[N/2] = in[0] - in[step];
    }else if(N == 4){
                fft(in, out, log2step+1, step *2 , N /2);
                fft(in+step, out+N/2, log2step+1, step *2, N /2);

                fcomplex temp0 = out[0] + out[2];
                fcomplex temp1 = out[0] - out[2];
                fcomplex temp2 = out[1] - I*out[3];
                fcomplex temp3 = out[1] + I*out[3];
                if(log2step) {
                    out[0] = creal(temp0) + creal(temp2)*I;
                    out[1] = creal(temp1) + creal(temp3)*I;
                    out[2] = cimag(temp0) + cimag(temp2)*I;
                    out[3] = cimag(temp1) + cimag(temp3)*I;
                }else{
                    out[0] = temp0;
                    out[1] = temp2;
                    out[2] = temp1;
                    out[3] = temp3;
                    }
                }
    else if(!log2step){
        fft(in, out, log2step+1, step * 2, N/2);
        fft(in+step, out+N/2, log2step+1, step*2 , N/2);

        int k;
        for(k=0;k<N/2;k+=4) {
            __m128 Ok_re = _mm_load_ps((float *)&out[k+N/2]);
            __m128 Ok_im = _mm_load_ps((float *)&out[k+N/2+2]);

            __m128 w_re = _mm_load_ps((float *)&twiddle[log2step][k]);
            __m128 w_im = _mm_load_ps((float *)&twiddle[log2step][k+2]);









            __m128 Ek_re = _mm_load_ps((float *)&out[k]);
            __m128 Ek_im = _mm_load_ps((float *)&out[k+2]);

            __m128 wOk_re = _mm_sub_ps(_mm_mul_ps(Ok_re,w_re),_mm_mul_ps(Ok_im,w_im));
            __m128 wOk_im = _mm_add_ps(_mm_mul_ps(Ok_re,w_im),_mm_mul_ps(Ok_im,w_re));

            __m128 out0_re = _mm_add_ps(Ek_re, wOk_re);
            __m128 out0_im = _mm_add_ps(Ek_im, wOk_im);
            __m128 out1_re = _mm_sub_ps(Ek_re, wOk_re);
            __m128 out1_im = _mm_sub_ps(Ek_im, wOk_im);

            _mm_store_ps((float *)(out+k), _mm_unpacklo_ps(out0_re, out0_im));
            _mm_store_ps((float *)(out+k+2), _mm_unpackhi_ps(out0_re, out0_im));
            _mm_store_ps((float *)(out+k+N/2), _mm_unpacklo_ps(out1_re, out1_im));
            _mm_store_ps((float *)(out+k+N/2+2), _mm_unpackhi_ps(out1_re, out1_im));
            }
        }
    else{
        fft(in, out, log2step+1, step *2, N/2);
        fft(in+step, out+N/2, log2step+1, step*2, N/2);

        int k;
        for(k=0;k<N/2;k+=4) {
            __m128 Ok_re = _mm_load_ps((float *)&out[k+N/2]);
            __m128 Ok_im = _mm_load_ps((float *)&out[k+N/2+2]);

            __m128 w_re = _mm_load_ps((float *)&twiddle[log2step][k]);
            __m128 w_im = _mm_load_ps((float *)&twiddle[log2step][k+2]);

            __m128 Ek_re = _mm_load_ps((float *)&out[k]);
            __m128 Ek_im = _mm_load_ps((float *)&out[k+2]);

            __m128 wOk_re = _mm_sub_ps(_mm_mul_ps(Ok_re,w_re),_mm_mul_ps(Ok_im,w_im));
            __m128 wOk_im = _mm_add_ps(_mm_mul_ps(Ok_re,w_im),_mm_mul_ps(Ok_im,w_re));

            _mm_store_ps((float *)(out+k), _mm_add_ps(Ek_re, wOk_re));
            _mm_store_ps((float *)(out+k+2), _mm_add_ps(Ek_im, wOk_im));
            _mm_store_ps((float *)(out+k+N/2), _mm_sub_ps(Ek_re, wOk_re));
            _mm_store_ps((float *)(out+k+N/2+2), _mm_sub_ps(Ek_im, wOk_im));
            }
        }
 }
