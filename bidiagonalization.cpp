// bidiagonalization.cpp
// 将 m×n 矩阵（本框架保证m ≥ n）通过 Householder 变换化为上双对角形
//
// 算法说明（你需要结合代码看）：
// 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// 第 k 步（k = 0, 1, ..., n-1）：
//    - 从左侧作用 H_k，消去第 k 列中位置 (k+1,k), (k+2,k), ..., (m-1,k) 的元素
//    - 如果 k < n-2，从右侧作用 V_k，消去第 k 行中位置 (k,k+2), (k,k+3), ..., (k,n-1) 的元素
//
// 例如，对一个 4x4 矩阵 A，第一步 k=0：
//   - 从左侧作用 H_0，消去 A(1,0), A(2,0), A(3,0)，得到 B_0，同时更新 U = U * H_0
//   - 从右侧作用 V_0，消去 B_0(0,2)，B_0(0,3)，得到 B_1，同时更新 V = V * V_0
//
// 最终得到上双对角矩阵 B，只有主对角线和上次对角线有非零元素
//
// 本组件输出：A = U * B * V^T
// 其中 U（m×m）和 V（n×n）均为正交矩阵，B（m×n）为上双对角矩阵

#include "matrix.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <arm_neon.h>
// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
{
    double sum = 0.0;
    int size=v.size();
    float64x2_t sum_data0=vdupq_n_f64(0.0);
    float64x2_t sum_data1=vdupq_n_f64(0.0);
    float64x2_t sum_data2=vdupq_n_f64(0.0);
    float64x2_t sum_data3=vdupq_n_f64(0.0);
    int i=0;
    for(;i+7<size;i+=8){
        float64x2_t data0=vld1q_f64(&v[i]);
        float64x2_t data1=vld1q_f64(&v[i+2]);
        float64x2_t data2=vld1q_f64(&v[i+4]);
        float64x2_t data3=vld1q_f64(&v[i+6]);
        sum_data0=vfmaq_f64(sum_data0,data0,data0);//平方和
        sum_data1=vfmaq_f64(sum_data1,data1,data1);
        sum_data2=vfmaq_f64(sum_data2,data2,data2);
        sum_data3=vfmaq_f64(sum_data3,data3,data3);
    }
    for(;i+3<size;i+=4){
        float64x2_t data0=vld1q_f64(&v[i]);
        float64x2_t data1=vld1q_f64(&v[i+2]);
        sum_data0=vfmaq_f64(sum_data0,data0,data0);//平方和
        sum_data1=vfmaq_f64(sum_data1,data1,data1);
    }
    for(;i+1<size;i+=2){    
        float64x2_t data=vld1q_f64(&v[i]);
        sum_data0=vfmaq_f64(sum_data0,data,data);//平方和
    }
    for(;i<size;i++){
        sum+=v[i]*v[i];
    }
    float64x2_t sum_data01=vaddq_f64(sum_data0,sum_data1);
    float64x2_t sum_data23=vaddq_f64(sum_data2,sum_data3);
    float64x2_t sum_data=vaddq_f64(sum_data01,sum_data23);
    sum+=vaddvq_f64(sum_data);
    return std::sqrt(sum);
}

// 将 m×n 矩阵 A（m ≥ n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    // U = I_m，V = I_n
    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
        U.at(i, i) = 1.0;
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
        V.at(i, i) = 1.0;

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1) 【ISSUE？长度是 m-k 吧】
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
        {
            x[i] = B.at(k + i, k);
        }

        double norm_x = vector_norm(x);//求二范数/模方

        if (norm_x > 1e-14 && k < m - 1)
        {
            // sign(x[0])：此处规定 x[0]==0 时取 +1
            //公式：v=x-αe_1(α = norm_x 符号下面定了，所以是)
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            // 实际上这里是+或者-都可以，手册里 Householder 一节是 -αe_1
            // 但我们这里 sigma 取了 sign(x[0]) * norm_x，所以是 +sigma * e_1 的形式
            std::vector<double> v(x); //+-e_1 相当于只有第一个元素改变值
            v[0] += sigma; // v = x + sigma * e_1

            // 计算 v^T v
            double vTv=0.0;
            int size_v=v.size();
            int i=0;
            float64x2_t sum_data0=vdupq_n_f64(0.0);
            float64x2_t sum_data1=vdupq_n_f64(0.0);
            float64x2_t sum_data2=vdupq_n_f64(0.0);
            float64x2_t sum_data3=vdupq_n_f64(0.0);
            for(;i+7<size_v;i+=8){
                float64x2_t data0=vld1q_f64(&v[i]);
                float64x2_t data1=vld1q_f64(&v[i+2]);
                float64x2_t data2=vld1q_f64(&v[i+4]);
                float64x2_t data3=vld1q_f64(&v[i+6]);
                sum_data0=vfmaq_f64(sum_data0,data0,data0);
                sum_data1=vfmaq_f64(sum_data1,data1,data1);
                sum_data2=vfmaq_f64(sum_data2,data2,data2);
                sum_data3=vfmaq_f64(sum_data3,data3,data3);
            }
            for(;i+3<size_v;i+=4){
                float64x2_t data0=vld1q_f64(&v[i]);
                float64x2_t data1=vld1q_f64(&v[i+2]);
                sum_data0=vfmaq_f64(sum_data0,data0,data0);
                sum_data1=vfmaq_f64(sum_data1,data1,data1);
            }
            for(;i+1<size_v;i+=2){
                float64x2_t data=vld1q_f64(&v[i]);
                sum_data0=vfmaq_f64(sum_data0,data,data);
            }
            for(;i<size_v;i++){
                vTv+=v[i]*v[i];
            }
            float64x2_t sum_data01=vaddq_f64(sum_data0,sum_data1);
            float64x2_t sum_data23=vaddq_f64(sum_data2,sum_data3);
            float64x2_t sum_data=vaddq_f64(sum_data01,sum_data23);
            vTv+=vaddvq_f64(sum_data);

            // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
                // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
                std::vector<double> w(n - k, 0.0);//存储结果
                //现在这样是列访问没办法连续，所以调一下顺序再 SIMD
                for(int i=0;i<m-k;i++){
                    float64x2_t v_data=vdupq_n_f64(v[i]);//拓宽
                    int j=0;
                    //>8
                    for(;j+7<n-k;j+=8){
                        float64x2_t b_data0=vld1q_f64(&B.at(k+i,k+j));
                        float64x2_t b_data1=vld1q_f64(&B.at(k+i,k+j+2));
                        float64x2_t b_data2=vld1q_f64(&B.at(k+i,k+j+4));
                        float64x2_t b_data3=vld1q_f64(&B.at(k+i,k+j+6));
                        float64x2_t w_data0=vld1q_f64(&w[j]);
                        float64x2_t w_data1=vld1q_f64(&w[j+2]);
                        float64x2_t w_data2=vld1q_f64(&w[j+4]);
                        float64x2_t w_data3=vld1q_f64(&w[j+6]);
                        w_data0=vaddq_f64(w_data0,vmulq_f64(v_data,b_data0));//乘加
                        w_data1=vaddq_f64(w_data1,vmulq_f64(v_data,b_data1));//乘加
                        w_data2=vaddq_f64(w_data2,vmulq_f64(v_data,b_data2));//乘加
                        w_data3=vaddq_f64(w_data3,vmulq_f64(v_data,b_data3));//乘加
                        vst1q_f64(&w[j],w_data0);//store
                        vst1q_f64(&w[j+2],w_data1);//store
                        vst1q_f64(&w[j+4],w_data2);//store
                        vst1q_f64(&w[j+6],w_data3);//store
                    }
                    //>4
                    for(;j+3<n-k;j+=4){
                        float64x2_t b_data0=vld1q_f64(&B.at(k+i,k+j));
                        float64x2_t b_data1=vld1q_f64(&B.at(k+i,k+j+2));
                        float64x2_t w_data0=vld1q_f64(&w[j]);
                        float64x2_t w_data1=vld1q_f64(&w[j+2]);
                        w_data0=vaddq_f64(w_data0,vmulq_f64(v_data,b_data0));//乘加
                        w_data1=vaddq_f64(w_data1,vmulq_f64(v_data,b_data1));//乘加
                        vst1q_f64(&w[j],w_data0);//store
                        vst1q_f64(&w[j+2],w_data1);//store
                    }
                    //>2
                    for(;j+1<n-k;j+=2){
                        float64x2_t b_data=vld1q_f64(&B.at(k+i,k+j));
                        float64x2_t w_data=vld1q_f64(&w[j]);
                        w_data=vaddq_f64(w_data,vmulq_f64(v_data,b_data));//乘加
                        vst1q_f64(&w[j],w_data);//store
                    }
                    //>1
                    for(;j<n-k;j++){
                        w[j]+=v[i]*B.at(k+i,k+j);
                        //v^T * B_old 因为算的是 (k,k)为左下角的矩阵，所以+k
                    }
                }
                //这个可以直接 SIMD 加速
                for (int i = 0; i < m - k; ++i){
                    float64x2_t v_data=vdupq_n_f64(beta*v[i]);//拓宽
                    int j=0;
                    for (;j+7<n-k;j+=8){
                        float64x2_t w_data0=vld1q_f64(&w[j]);
                        float64x2_t w_data1=vld1q_f64(&w[j+2]);
                        float64x2_t w_data2=vld1q_f64(&w[j+4]);
                        float64x2_t w_data3=vld1q_f64(&w[j+6]);
                        float64x2_t new_data0=vmulq_f64(v_data,w_data0);//multi
                        float64x2_t new_data1=vmulq_f64(v_data,w_data1);//multi
                        float64x2_t new_data2=vmulq_f64(v_data,w_data2);//multi
                        float64x2_t new_data3=vmulq_f64(v_data,w_data3);//multi
                        float64x2_t b_data0=vld1q_f64(&B.at(k+i,k+j));
                        float64x2_t b_data1=vld1q_f64(&B.at(k+i,k+j+2));
                        float64x2_t b_data2=vld1q_f64(&B.at(k+i,k+j+4));
                        float64x2_t b_data3=vld1q_f64(&B.at(k+i,k+j+6));
                        b_data0=vsubq_f64(b_data0,new_data0);//update
                        b_data1=vsubq_f64(b_data1,new_data1);//update
                        b_data2=vsubq_f64(b_data2,new_data2);//update
                        b_data3=vsubq_f64(b_data3,new_data3);//update
                        vst1q_f64(&B.at(k+i,k+j),b_data0);//store
                        vst1q_f64(&B.at(k+i,k+j+2),b_data1);//store
                        vst1q_f64(&B.at(k+i,k+j+4),b_data2);//store
                        vst1q_f64(&B.at(k+i,k+j+6),b_data3);//store
                    }
                    for (;j+3<n-k;j+=4){
                        float64x2_t w_data0=vld1q_f64(&w[j]);
                        float64x2_t w_data1=vld1q_f64(&w[j+2]);
                        float64x2_t new_data0=vmulq_f64(v_data,w_data0);//multi
                        float64x2_t new_data1=vmulq_f64(v_data,w_data1);//multi
                        float64x2_t b_data0=vld1q_f64(&B.at(k+i,k+j));
                        float64x2_t b_data1=vld1q_f64(&B.at(k+i,k+j+2));
                        b_data0=vsubq_f64(b_data0,new_data0);//update
                        b_data1=vsubq_f64(b_data1,new_data1);//update
                        vst1q_f64(&B.at(k+i,k+j),b_data0);//store
                        vst1q_f64(&B.at(k+i,k+j+2),b_data1);//store
                    }
                    for (;j+1<n-k;j+=2){
                        float64x2_t w_data=vld1q_f64(&w[j]);
                        float64x2_t new_data=vmulq_f64(v_data,w_data);//multi
                        float64x2_t b_data=vld1q_f64(&B.at(k+i,k+j));
                        b_data=vsubq_f64(b_data,new_data);//update
                        vst1q_f64(&B.at(k+i,k+j),b_data);//store
                    }
                    for(;j<n-k;j++){
                        B.at(k + i, k + j) -= beta * v[i] * w[j];
                    }
                }

                // 累积 U：U_new = U_old * H_k
                // U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                std::vector<double> wU(m, 0.0);
                for (int i = 0; i < m; ++i){
                    int j = 0;
                    float64x2_t sum_v0=vdupq_n_f64(0.0);
                    float64x2_t sum_v1=vdupq_n_f64(0.0);
                    float64x2_t sum_v2=vdupq_n_f64(0.0);
                    float64x2_t sum_v3=vdupq_n_f64(0.0);
                    for (;j+7<m-k; j+=8){
                        float64x2_t u_data0=vld1q_f64(&U.at(i,k+j));
                        float64x2_t u_data1=vld1q_f64(&U.at(i,k+j+2));
                        float64x2_t u_data2=vld1q_f64(&U.at(i,k+j+4));
                        float64x2_t u_data3=vld1q_f64(&U.at(i,k+j+6));
                        float64x2_t v_data0=vld1q_f64(&v[j]);
                        float64x2_t v_data1=vld1q_f64(&v[j+2]);
                        float64x2_t v_data2=vld1q_f64(&v[j+4]);
                        float64x2_t v_data3=vld1q_f64(&v[j+6]);
                        sum_v0=vaddq_f64(sum_v0,vmulq_f64(u_data0,v_data0));//multi
                        sum_v1=vaddq_f64(sum_v1,vmulq_f64(u_data1,v_data1));//multi
                        sum_v2=vaddq_f64(sum_v2,vmulq_f64(u_data2,v_data2));//multi
                        sum_v3=vaddq_f64(sum_v3,vmulq_f64(u_data3,v_data3));//multi
                    }
                    for (;j+3<m-k; j+=4){
                        float64x2_t u_data0=vld1q_f64(&U.at(i,k+j));
                        float64x2_t u_data1=vld1q_f64(&U.at(i,k+j+2));
                        float64x2_t v_data0=vld1q_f64(&v[j]);
                        float64x2_t v_data1=vld1q_f64(&v[j+2]);
                        sum_v0=vaddq_f64(sum_v0,vmulq_f64(u_data0,v_data0));//multi
                        sum_v1=vaddq_f64(sum_v1,vmulq_f64(u_data1,v_data1));//multi
                    }
                    for (;j+1<m-k; j+=2){
                        float64x2_t u_data=vld1q_f64(&U.at(i,k+j));
                        float64x2_t v_data=vld1q_f64(&v[j]);
                        sum_v0=vaddq_f64(sum_v0,vmulq_f64(u_data,v_data));//multi
                    }
                    double tmp0[2];
                    double tmp1[2];
                    double tmp2[2];
                    double tmp3[2];
                    vst1q_f64(tmp0, sum_v0);
                    vst1q_f64(tmp1, sum_v1);
                    vst1q_f64(tmp2, sum_v2);
                    vst1q_f64(tmp3, sum_v3);
                    double sum=tmp0[0]+tmp0[1]+tmp1[0]+tmp1[1]+tmp2[0]+tmp2[1]+tmp3[0]+tmp3[1];
                    for(;j<m-k;j++)
                        sum+= U.at(i, k + j) * v[j];
                    wU[i]=sum;
                }
                for (int i = 0; i < m; ++i){
                    int j = 0;
                    float64x2_t w_data=vdupq_n_f64(beta*wU[i]);//拓宽
                    for (; j+7<m-k;j+=8){
                        float64x2_t v_data0=vld1q_f64(&v[j]);
                        float64x2_t v_data1=vld1q_f64(&v[j+2]);
                        float64x2_t v_data2=vld1q_f64(&v[j+4]);
                        float64x2_t v_data3=vld1q_f64(&v[j+6]);
                        float64x2_t new_data0=vmulq_f64(w_data,v_data0);//multi
                        float64x2_t new_data1=vmulq_f64(w_data,v_data1);//multi
                        float64x2_t new_data2=vmulq_f64(w_data,v_data2);//multi
                        float64x2_t new_data3=vmulq_f64(w_data,v_data3);//multi
                        float64x2_t u_data0=vld1q_f64(&U.at(i,k+j));
                        float64x2_t u_data1=vld1q_f64(&U.at(i,k+j+2));
                        float64x2_t u_data2=vld1q_f64(&U.at(i,k+j+4));
                        float64x2_t u_data3=vld1q_f64(&U.at(i,k+j+6));
                        u_data0=vsubq_f64(u_data0,new_data0);//update
                        u_data1=vsubq_f64(u_data1,new_data1);//update
                        u_data2=vsubq_f64(u_data2,new_data2);//update
                        u_data3=vsubq_f64(u_data3,new_data3);//update
                        vst1q_f64(&U.at(i,k+j),u_data0);//store
                        vst1q_f64(&U.at(i,k+j+2),u_data1);//store
                        vst1q_f64(&U.at(i,k+j+4),u_data2);//store
                        vst1q_f64(&U.at(i,k+j+6),u_data3);//store
                    }
                    for (; j+3<m-k;j+=4){
                        float64x2_t v_data0=vld1q_f64(&v[j]);
                        float64x2_t v_data1=vld1q_f64(&v[j+2]);
                        float64x2_t new_data0=vmulq_f64(w_data,v_data0);//multi
                        float64x2_t new_data1=vmulq_f64(w_data,v_data1);//multi
                        float64x2_t u_data0=vld1q_f64(&U.at(i,k+j));
                        float64x2_t u_data1=vld1q_f64(&U.at(i,k+j+2));
                        u_data0=vsubq_f64(u_data0,new_data0);//update
                        u_data1=vsubq_f64(u_data1,new_data1);//update
                        vst1q_f64(&U.at(i,k+j),u_data0);//store
                        vst1q_f64(&U.at(i,k+j+2),u_data1);//store
                    }
                    for (; j+1<m - k;j+=2){
                        float64x2_t v_data=vld1q_f64(&v[j]);
                        float64x2_t new_data=vmulq_f64(w_data,v_data);//multi
                        float64x2_t u_data=vld1q_f64(&U.at(i,k+j));
                        u_data=vsubq_f64(u_data,new_data);//update
                        vst1q_f64(&U.at(i,k+j),u_data);//store
                    }
                    for(;j<m-k;j++)
                        U.at(i, k + j) -= beta * wU[i] * v[j];
                }
            }
        }

        // 清除第 k 列中对角线以下的元素
        // 理论上应为 0，但不能完全保证全是 0，这里强制置零
        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        //        （只在 k < n-2 时需要）
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
            {
                y[j] = B.at(k, k + 1 + j);
            }

            // 与之前类似，计算模长
            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                // 构造 Householder 向量 v = y + sigma * e_1
                std::vector<double> v(y);
                v[0] += sigma;

                double vTv=0.0;
                int size_v=v.size();
                int i=0;
                float64x2_t sum_data0=vdupq_n_f64(0.0);
                float64x2_t sum_data1=vdupq_n_f64(0.0);
                float64x2_t sum_data2=vdupq_n_f64(0.0);
                float64x2_t sum_data3=vdupq_n_f64(0.0);
                for(;i+7<size_v;i+=8){
                    float64x2_t data0=vld1q_f64(&v[i]);
                    float64x2_t data1=vld1q_f64(&v[i+2]);
                    float64x2_t data2=vld1q_f64(&v[i+4]);
                    float64x2_t data3=vld1q_f64(&v[i+6]);
                    sum_data0=vfmaq_f64(sum_data0,data0,data0);
                    sum_data1=vfmaq_f64(sum_data1,data1,data1);
                    sum_data2=vfmaq_f64(sum_data2,data2,data2);
                    sum_data3=vfmaq_f64(sum_data3,data3,data3);
                }
                for(;i+3<size_v;i+=4){
                    float64x2_t data0=vld1q_f64(&v[i]);
                    float64x2_t data1=vld1q_f64(&v[i+2]);
                    sum_data0=vfmaq_f64(sum_data0,data0,data0);
                    sum_data1=vfmaq_f64(sum_data1,data1,data1);
                }
                for(;i+1<size_v;i+=2){
                    float64x2_t data=vld1q_f64(&v[i]);
                    sum_data0=vfmaq_f64(sum_data0,data,data);
                }
                for(;i<size_v;i++){
                    vTv+=v[i]*v[i];
                }
                float64x2_t sum_data01=vaddq_f64(sum_data0,sum_data1);
                float64x2_t sum_data23=vaddq_f64(sum_data2,sum_data3);
                float64x2_t sum_data=vaddq_f64(sum_data01,sum_data23);
                vTv+=vaddvq_f64(sum_data);


                // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 注意：这里是从右侧作用 V_k
                    // B_new = B_old * V_k = B_old - beta * (B_old * v) * v^T
                    std::vector<double> w(m - k, 0.0);
                    //现在这样本来就是按行连续访问，可以直接 SIMD
                    for (int i = 0; i < m - k; ++i){
                        int j=0;
                        float64x2_t sum_v0=vdupq_n_f64(0.0);
                        float64x2_t sum_v1=vdupq_n_f64(0.0);
                        float64x2_t sum_v2=vdupq_n_f64(0.0);
                        float64x2_t sum_v3=vdupq_n_f64(0.0);
                        for (;j+7<n-k-1;j+=8){
                            float64x2_t b_data0=vld1q_f64(&B.at(k+i,k+1+j));
                            float64x2_t b_data1=vld1q_f64(&B.at(k+i,k+1+j+2));
                            float64x2_t b_data2=vld1q_f64(&B.at(k+i,k+1+j+4));
                            float64x2_t b_data3=vld1q_f64(&B.at(k+i,k+1+j+6));
                            float64x2_t v_data0=vld1q_f64(&v[j]);
                            float64x2_t v_data1=vld1q_f64(&v[j+2]);
                            float64x2_t v_data2=vld1q_f64(&v[j+4]);
                            float64x2_t v_data3=vld1q_f64(&v[j+6]);
                            sum_v0=vaddq_f64(sum_v0,vmulq_f64(b_data0,v_data0));//multi
                            sum_v1=vaddq_f64(sum_v1,vmulq_f64(b_data1,v_data1));//multi
                            sum_v2=vaddq_f64(sum_v2,vmulq_f64(b_data2,v_data2));//multi
                            sum_v3=vaddq_f64(sum_v3,vmulq_f64(b_data3,v_data3));//multi
                        }
                        for (;j+3<n-k-1;j+=4){
                            float64x2_t b_data0=vld1q_f64(&B.at(k+i,k+1+j));
                            float64x2_t b_data1=vld1q_f64(&B.at(k+i,k+1+j+2));
                            float64x2_t v_data0=vld1q_f64(&v[j]);
                            float64x2_t v_data1=vld1q_f64(&v[j+2]);
                            sum_v0=vaddq_f64(sum_v0,vmulq_f64(b_data0,v_data0));//multi
                            sum_v1=vaddq_f64(sum_v1,vmulq_f64(b_data1,v_data1));//multi
                        }
                        for (;j+1<n-k-1;j+=2){
                            float64x2_t b_data=vld1q_f64(&B.at(k+i,k+1+j));
                            float64x2_t v_data=vld1q_f64(&v[j]);
                            sum_v0=vaddq_f64(sum_v0,vmulq_f64(b_data,v_data));//multi
                        }
                        double tmp0[2];
                        double tmp1[2];
                        double tmp2[2];
                        double tmp3[2];
                        vst1q_f64(tmp0,sum_v0);
                        vst1q_f64(tmp1,sum_v1);
                        vst1q_f64(tmp2,sum_v2);
                        vst1q_f64(tmp3,sum_v3);
                        double sum=tmp0[0]+tmp0[1]+tmp1[0]+tmp1[1]+tmp2[0]+tmp2[1]+tmp3[0]+tmp3[1];
                        for (;j<n-k-1;j++){
                            sum+=B.at(k+i,k+1+j)*v[j];
                        }
                        w[i]=sum;
                    }
                    for (int i = 0; i < m - k; ++i){
                        int j=0;
                        float64x2_t w_data=vdupq_n_f64(beta*w[i]);//拓宽
                        for (;j+7<n-k-1;j+=8){
                            float64x2_t v_data0=vld1q_f64(&v[j]);
                            float64x2_t v_data1=vld1q_f64(&v[j+2]);
                            float64x2_t v_data2=vld1q_f64(&v[j+4]);
                            float64x2_t v_data3=vld1q_f64(&v[j+6]);
                            float64x2_t new_data0=vmulq_f64(w_data,v_data0);//multi
                            float64x2_t new_data1=vmulq_f64(w_data,v_data1);//multi
                            float64x2_t new_data2=vmulq_f64(w_data,v_data2);//multi
                            float64x2_t new_data3=vmulq_f64(w_data,v_data3);//multi
                            float64x2_t b_data0=vld1q_f64(&B.at(k+i,k+1+j));
                            float64x2_t b_data1=vld1q_f64(&B.at(k+i,k+1+j+2));
                            float64x2_t b_data2=vld1q_f64(&B.at(k+i,k+1+j+4));
                            float64x2_t b_data3=vld1q_f64(&B.at(k+i,k+1+j+6));
                            b_data0=vsubq_f64(b_data0,new_data0);//update
                            b_data1=vsubq_f64(b_data1,new_data1);//update
                            b_data2=vsubq_f64(b_data2,new_data2);//update
                            b_data3=vsubq_f64(b_data3,new_data3);//update
                            vst1q_f64(&B.at(k+i,k+1+j),b_data0);//store
                            vst1q_f64(&B.at(k+i,k+1+j+2),b_data1);//store
                            vst1q_f64(&B.at(k+i,k+1+j+4),b_data2);//store
                            vst1q_f64(&B.at(k+i,k+1+j+6),b_data3);//store
                        }
                        for (;j+3<n-k-1;j+=4){
                            float64x2_t v_data0=vld1q_f64(&v[j]);
                            float64x2_t v_data1=vld1q_f64(&v[j+2]);
                            float64x2_t new_data0=vmulq_f64(w_data,v_data0);//multi
                            float64x2_t new_data1=vmulq_f64(w_data,v_data1);//multi
                            float64x2_t b_data0=vld1q_f64(&B.at(k+i,k+1+j));
                            float64x2_t b_data1=vld1q_f64(&B.at(k+i,k+1+j+2));
                            b_data0=vsubq_f64(b_data0,new_data0);//update
                            b_data1=vsubq_f64(b_data1,new_data1);//update
                            vst1q_f64(&B.at(k+i,k+1+j),b_data0);//store
                            vst1q_f64(&B.at(k+i,k+1+j+2),b_data1);//store
                        }
                        for (;j+1<n-k-1;j+=2){
                            float64x2_t v_data=vld1q_f64(&v[j]);
                            float64x2_t new_data=vmulq_f64(w_data,v_data);//multi
                            float64x2_t b_data=vld1q_f64(&B.at(k+i,k+1+j));
                            b_data=vsubq_f64(b_data,new_data);//update
                            vst1q_f64(&B.at(k+i,k+1+j),b_data);//store
                        }
                        for (;j<n-k-1;j++){
                            B.at(k+i,k+1+j)-=beta*w[i]*v[j];
                        }
                    }

                    // 累积 V：V_new = V_old * V_k
                    // V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
                    std::vector<double> wV(n, 0.0);

                    //这个也是按行连续访问，可以 SIMD 求每一行和 v 的点积
                    for (int i = 0; i < n; ++i){
                        int j=0;
                        float64x2_t sum_v0=vdupq_n_f64(0.0);
                        float64x2_t sum_v1=vdupq_n_f64(0.0);
                        float64x2_t sum_v2=vdupq_n_f64(0.0);
                        float64x2_t sum_v3=vdupq_n_f64(0.0);
                        for (;j+7<n-k-1;j+=8){
                            float64x2_t V_data0=vld1q_f64(&V.at(i,k+1+j));
                            float64x2_t V_data1=vld1q_f64(&V.at(i,k+1+j+2));
                            float64x2_t V_data2=vld1q_f64(&V.at(i,k+1+j+4));
                            float64x2_t V_data3=vld1q_f64(&V.at(i,k+1+j+6));
                            float64x2_t v_data0=vld1q_f64(&v[j]);
                            float64x2_t v_data1=vld1q_f64(&v[j+2]);
                            float64x2_t v_data2=vld1q_f64(&v[j+4]);
                            float64x2_t v_data3=vld1q_f64(&v[j+6]);
                            sum_v0=vaddq_f64(sum_v0,vmulq_f64(V_data0,v_data0));//multi
                            sum_v1=vaddq_f64(sum_v1,vmulq_f64(V_data1,v_data1));//multi
                            sum_v2=vaddq_f64(sum_v2,vmulq_f64(V_data2,v_data2));//multi
                            sum_v3=vaddq_f64(sum_v3,vmulq_f64(V_data3,v_data3));//multi
                        }
                        for (;j+3<n-k-1;j+=4){
                            float64x2_t V_data0=vld1q_f64(&V.at(i,k+1+j));
                            float64x2_t V_data1=vld1q_f64(&V.at(i,k+1+j+2));
                            float64x2_t v_data0=vld1q_f64(&v[j]);
                            float64x2_t v_data1=vld1q_f64(&v[j+2]);
                            sum_v0=vaddq_f64(sum_v0,vmulq_f64(V_data0,v_data0));//multi
                            sum_v1=vaddq_f64(sum_v1,vmulq_f64(V_data1,v_data1));//multi
                        }
                        for (;j+1<n-k-1;j+=2){
                            float64x2_t v_data=vld1q_f64(&v[j]);
                            float64x2_t V_data=vld1q_f64(&V.at(i,k+1+j));
                            sum_v0=vaddq_f64(sum_v0,vmulq_f64(V_data,v_data));//multi
                        }
                        double tmp0[2];
                        double tmp1[2];
                        double tmp2[2];
                        double tmp3[2];
                        vst1q_f64(tmp0,sum_v0);
                        vst1q_f64(tmp1,sum_v1);
                        vst1q_f64(tmp2,sum_v2);
                        vst1q_f64(tmp3,sum_v3);
                        double sum=tmp0[0]+tmp0[1]+tmp1[0]+tmp1[1]+tmp2[0]+tmp2[1]+tmp3[0]+tmp3[1];
                        for (;j<n-k-1;j++){
                            sum+=V.at(i,k+1+j)*v[j];
                        }
                        wV[i]=sum;
                    }
                    //这个可以直接 SIMD 加速
                    for (int i = 0; i < n; ++i){
                        int j=0;
                        float64x2_t w_data=vdupq_n_f64(beta*wV[i]);//拓宽
                        for (;j+7<n-k-1;j+=8){
                            float64x2_t v_data0=vld1q_f64(&v[j]);
                            float64x2_t v_data1=vld1q_f64(&v[j+2]);
                            float64x2_t v_data2=vld1q_f64(&v[j+4]);
                            float64x2_t v_data3=vld1q_f64(&v[j+6]);
                            float64x2_t new_data0=vmulq_f64(w_data,v_data0);//multi
                            float64x2_t new_data1=vmulq_f64(w_data,v_data1);//multi
                            float64x2_t new_data2=vmulq_f64(w_data,v_data2);//multi
                            float64x2_t new_data3=vmulq_f64(w_data,v_data3);//multi
                            float64x2_t V_data0=vld1q_f64(&V.at(i,k+1+j));
                            float64x2_t V_data1=vld1q_f64(&V.at(i,k+1+j+2));
                            float64x2_t V_data2=vld1q_f64(&V.at(i,k+1+j+4));
                            float64x2_t V_data3=vld1q_f64(&V.at(i,k+1+j+6));
                            V_data0=vsubq_f64(V_data0,new_data0);//update
                            V_data1=vsubq_f64(V_data1,new_data1);//update
                            V_data2=vsubq_f64(V_data2,new_data2);//update
                            V_data3=vsubq_f64(V_data3,new_data3);//update
                            vst1q_f64(&V.at(i,k+1+j),V_data0);//store
                            vst1q_f64(&V.at(i,k+1+j+2),V_data1);//store
                            vst1q_f64(&V.at(i,k+1+j+4),V_data2);//store
                            vst1q_f64(&V.at(i,k+1+j+6),V_data3);//store
                        }
                        for (;j+3<n-k-1;j+=4){
                            float64x2_t v_data0=vld1q_f64(&v[j]);
                            float64x2_t v_data1=vld1q_f64(&v[j+2]);
                            float64x2_t new_data0=vmulq_f64(w_data,v_data0);//multi
                            float64x2_t new_data1=vmulq_f64(w_data,v_data1);//multi
                            float64x2_t V_data0=vld1q_f64(&V.at(i,k+1+j));
                            float64x2_t V_data1=vld1q_f64(&V.at(i,k+1+j+2));
                            V_data0=vsubq_f64(V_data0,new_data0);//update
                            V_data1=vsubq_f64(V_data1,new_data1);//update
                            vst1q_f64(&V.at(i,k+1+j),V_data0);//store
                            vst1q_f64(&V.at(i,k+1+j+2),V_data1);//store
                        }
                        for (;j+1<n-k-1;j+=2){
                            float64x2_t v_data=vld1q_f64(&v[j]);
                            float64x2_t new_data=vmulq_f64(w_data,v_data);//multi
                            float64x2_t V_data=vld1q_f64(&V.at(i,k+1+j));
                            V_data=vsubq_f64(V_data,new_data);//update
                            vst1q_f64(&V.at(i,k+1+j),V_data);//store
                        }
                        for (;j<n-k-1;j++){
                            V.at(i,k+1+j)-=beta*wV[i]*v[j];
                        }
                    }
                }
            }

            // 强制置零
            for (int j = k + 2; j < n; ++j)
            {
                B.at(k, j) = 0.0;
            }
        }
    }

    return B;
}
