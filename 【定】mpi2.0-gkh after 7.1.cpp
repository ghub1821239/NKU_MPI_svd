#include "gkh.h"
#include "givens.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>
#include<iostream>
#include <mpi.h>

namespace
{
    struct TimingStats
    {
        double init_scatter_ms = 0.0;
        double bcast_ctrl_ms = 0.0;
        double bcast_uops_ms = 0.0;
        double bcast_vops_ms = 0.0;
        double apply_ut_ms = 0.0;
        double apply_vt_ms = 0.0;
        double final_gather_ms = 0.0;
        double root_extract_insert_ms = 0.0;
        double root_process_ms = 0.0;
        double root_finalize_ms = 0.0;
        double total_wall_ms = 0.0;

        unsigned long long logical_scatter_bytes = 0;
        unsigned long long logical_bcast_bytes = 0;
        unsigned long long logical_gather_bytes = 0;
        unsigned long long iterations = 0;
        unsigned long long u_ops_total = 0;
        unsigned long long v_ops_total = 0;
    };

    static TimingStats *g_stats = nullptr;
    static int g_bcast_kind = 0;

    static double elapsed_ms(double t0, double t1)
    {
        return (t1 - t0) * 1000.0;
    }

    static void reduce_sum_max(double local, double &sum, double &max)
    {
        MPI_Reduce(&local, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local, &max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    static void reduce_sum_ull(unsigned long long local, unsigned long long &sum)
    {
        MPI_Reduce(&local, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    static void print_timing_summary_mpi2(const TimingStats &local, int rank)
    {
        double init_scatter_sum = 0.0, init_scatter_max = 0.0;
        double bcast_ctrl_sum = 0.0, bcast_ctrl_max = 0.0;
        double bcast_uops_sum = 0.0, bcast_uops_max = 0.0;
        double bcast_vops_sum = 0.0, bcast_vops_max = 0.0;
        double apply_ut_sum = 0.0, apply_ut_max = 0.0;
        double apply_vt_sum = 0.0, apply_vt_max = 0.0;
        double final_gather_sum = 0.0, final_gather_max = 0.0;
        double total_wall_max = 0.0;

        reduce_sum_max(local.init_scatter_ms, init_scatter_sum, init_scatter_max);
        reduce_sum_max(local.bcast_ctrl_ms, bcast_ctrl_sum, bcast_ctrl_max);
        reduce_sum_max(local.bcast_uops_ms, bcast_uops_sum, bcast_uops_max);
        reduce_sum_max(local.bcast_vops_ms, bcast_vops_sum, bcast_vops_max);
        reduce_sum_max(local.apply_ut_ms, apply_ut_sum, apply_ut_max);
        reduce_sum_max(local.apply_vt_ms, apply_vt_sum, apply_vt_max);
        reduce_sum_max(local.final_gather_ms, final_gather_sum, final_gather_max);
        MPI_Reduce(&local.total_wall_ms, &total_wall_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        unsigned long long scatter_bytes = 0;
        unsigned long long bcast_bytes = 0;
        unsigned long long gather_bytes = 0;
        unsigned long long iterations = 0;
        unsigned long long u_ops_total = 0;
        unsigned long long v_ops_total = 0;
        reduce_sum_ull(local.logical_scatter_bytes, scatter_bytes);
        reduce_sum_ull(local.logical_bcast_bytes, bcast_bytes);
        reduce_sum_ull(local.logical_gather_bytes, gather_bytes);
        reduce_sum_ull(local.iterations, iterations);
        reduce_sum_ull(local.u_ops_total, u_ops_total);
        reduce_sum_ull(local.v_ops_total, v_ops_total);

        if (rank != 0)
        {
            return;
        }

        const double comm_sum = init_scatter_sum + bcast_ctrl_sum + bcast_uops_sum +
                                bcast_vops_sum + final_gather_sum;
        const double update_sum = apply_ut_sum + apply_vt_sum;

        std::cout << "[mpi2.0-after7.1 timing]\n";
        std::cout << "  total wall(ms, MPI_MAX)   : " << total_wall_max << "\n";
        std::cout << "  root extract+insert(ms)   : " << local.root_extract_insert_ms << "\n";
        std::cout << "  root process block(ms)    : " << local.root_process_ms << "\n";
        std::cout << "  init scatter(ms)          : sum " << init_scatter_sum << ", max " << init_scatter_max << "\n";
        std::cout << "  bcast ctrl(ms)            : sum " << bcast_ctrl_sum << ", max " << bcast_ctrl_max << "\n";
        std::cout << "  bcast u_ops(ms)           : sum " << bcast_uops_sum << ", max " << bcast_uops_max << "\n";
        std::cout << "  bcast v_ops(ms)           : sum " << bcast_vops_sum << ", max " << bcast_vops_max << "\n";
        std::cout << "  apply Ut local(ms)        : sum " << apply_ut_sum << ", max " << apply_ut_max << "\n";
        std::cout << "  apply Vt local(ms)        : sum " << apply_vt_sum << ", max " << apply_vt_max << "\n";
        std::cout << "  final gather(ms)          : sum " << final_gather_sum << ", max " << final_gather_max << "\n";
        std::cout << "  root finalize(ms)         : " << local.root_finalize_ms << "\n";
        std::cout << "  comm subtotal(ms, sum)    : " << comm_sum << "\n";
        std::cout << "  local update(ms, sum)     : " << update_sum << "\n";
        std::cout << "  iterations                : " << iterations << "\n";
        std::cout << "  total u_ops / v_ops       : " << u_ops_total << " / " << v_ops_total << "\n";
        std::cout << "  logical payload bytes     : scatter " << scatter_bytes
                  << ", bcast " << bcast_bytes
                  << ", gather " << gather_bytes << "\n";
    }

    static Matrix transpose_copy(const Matrix &A)
    {
        Matrix T(A.cols(),A.rows(),0.0);
        for (int i=0;i<A.rows();++i){
            for (int j=0;j<A.cols();++j)
            {
                T.at(j,i)=A.at(i,j);
            }
        }
        return T;
    }

    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    // 在该区间内，超对角线元素非零，你可以认为通过这个抽象结构给矩阵“分块”。
    struct Block
    {
        int l;
        int r;
    };
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s);
    struct RowRotation
    {
        int r0;
        int r1;
        double c;
        double s;
    };

    static void column_block_bounds(int total_cols, int rank, int size, int &begin, int &end)
    {
        const int base = total_cols / size;
        const int rem = total_cols % size;
        begin = rank * base + std::min(rank, rem);
        end = begin + base + (rank < rem ? 1 : 0);
    }

    static Matrix extract_col_block(const Matrix &M, int begin, int end)
    {
        Matrix blk(M.rows(), end - begin, 0.0);
        for (int i = 0; i < M.rows(); ++i)
        {
            for (int j = begin; j < end; ++j)
            {
                blk.at(i, j - begin) = M.at(i, j);
            }
        }
        return blk;
    }

    static void insert_col_block(Matrix &M, const Matrix &blk, int begin)
    {
        for (int i = 0; i < M.rows(); ++i)
        {
            for (int j = 0; j < blk.cols(); ++j)
            {
                M.at(i, begin + j) = blk.at(i, j);
            }
        }
    }

    static std::vector<double> pack_matrix(const Matrix &M)
    {
        std::vector<double> buf(static_cast<size_t>(M.rows()) * static_cast<size_t>(M.cols()));
        size_t p = 0;
        for (int i = 0; i < M.rows(); ++i)
        {
            for (int j = 0; j < M.cols(); ++j)
            {
                buf[p++] = M.at(i, j);
            }
        }
        return buf;
    }

    static void unpack_matrix(Matrix &M, const std::vector<double> &buf)
    {
        size_t p = 0;
        for (int i = 0; i < M.rows(); ++i)
        {
            for (int j = 0; j < M.cols(); ++j)
            {
                M.at(i, j) = buf[p++];
            }
        }
    }

    static Matrix scatter_col_blocks_from_root(const Matrix &root_matrix,
                                               int full_rows,
                                               int full_cols,
                                               int root,
                                               MPI_Comm comm)
    {
        const double t_beg = MPI_Wtime();
        int rank = 0, size = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        int begin = 0, end = 0;
        column_block_bounds(full_cols, rank, size, begin, end);
        Matrix local_blk(full_rows, end - begin, 0.0);

        if (rank == root)
        {
            local_blk = extract_col_block(root_matrix, begin, end);

            for (int dst = 0; dst < size; ++dst)
            {
                if (dst == root)
                {
                    continue;
                }

                int b = 0, e = 0;
                column_block_bounds(full_cols, dst, size, b, e);
                Matrix blk = extract_col_block(root_matrix, b, e);
                std::vector<double> buf = pack_matrix(blk);
                if (!buf.empty())
                {
                    if (g_stats != nullptr)
                    {
                        g_stats->logical_scatter_bytes += static_cast<unsigned long long>(buf.size() * sizeof(double));
                    }
                    MPI_Send(buf.data(),
                             static_cast<int>(buf.size()),
                             MPI_DOUBLE,
                             dst,
                             0,
                             comm);
                }
            }

            if (g_stats != nullptr)
            {
                g_stats->init_scatter_ms += elapsed_ms(t_beg, MPI_Wtime());
            }
            return local_blk;
        }

        std::vector<double> buf(static_cast<size_t>(full_rows) * static_cast<size_t>(end - begin));
        if (!buf.empty())
        {
            MPI_Recv(buf.data(),
                     static_cast<int>(buf.size()),
                     MPI_DOUBLE,
                     root,
                     0,
                     comm,
                     MPI_STATUS_IGNORE);
            unpack_matrix(local_blk, buf);
        }

        if (g_stats != nullptr)
        {
            g_stats->init_scatter_ms += elapsed_ms(t_beg, MPI_Wtime());
        }
        return local_blk;
    }

    static void bcast_row_rotations(std::vector<RowRotation> &ops, int root, MPI_Comm comm)
    {
        const double t_beg = MPI_Wtime();
        int rank = 0;
        int size = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        int count = (rank == root) ? static_cast<int>(ops.size()) : 0;
        MPI_Bcast(&count, 1, MPI_INT, root, comm);

        if (rank != root)
        {
            ops.resize(count);
        }

        if (count > 0)
        {
            MPI_Bcast(
                reinterpret_cast<void *>(ops.data()),
                static_cast<int>(count * sizeof(RowRotation)),
                MPI_BYTE,
                root,
                comm
            );
        }

        if (g_stats != nullptr)
        {
            const double dt = elapsed_ms(t_beg, MPI_Wtime());
            if (g_bcast_kind == 1)
            {
                g_stats->bcast_uops_ms += dt;
            }
            else if (g_bcast_kind == 2)
            {
                g_stats->bcast_vops_ms += dt;
            }

            if (count > 0 && rank == root)
            {
                g_stats->logical_bcast_bytes += static_cast<unsigned long long>(count * sizeof(RowRotation)) *
                                                static_cast<unsigned long long>(size - 1);
                g_stats->logical_bcast_bytes += static_cast<unsigned long long>(sizeof(int)) *
                                                static_cast<unsigned long long>(size - 1);
            }
        }
    }

    static void apply_row_rotations_local(Matrix &M, const std::vector<RowRotation> &ops)
    {
        for (const auto &op : ops)
        {
            apply_left_rows(M, op.r0, op.r1, op.c, op.s);
        }
    }

    static Matrix gather_col_blocks_to_root(const Matrix &local_blk,
                                            int full_rows,
                                            int full_cols,
                                            int root,
                                            MPI_Comm comm)
    {
        const double t_beg = MPI_Wtime();
        int rank = 0, size = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        if (rank == root)
        {
            Matrix full(full_rows, full_cols, 0.0);

            int begin = 0, end = 0;
            column_block_bounds(full_cols, rank, size, begin, end);
            insert_col_block(full, local_blk, begin);

            for (int src = 1; src < size; ++src)
            {
                int b = 0, e = 0;
                column_block_bounds(full_cols, src, size, b, e);
                Matrix tmp(full_rows, e - b, 0.0);
                std::vector<double> buf(static_cast<size_t>(full_rows) * static_cast<size_t>(e - b));
                MPI_Recv(buf.data(),
                         static_cast<int>(buf.size()),
                         MPI_DOUBLE,
                         src,
                         0,
                         comm,
                         MPI_STATUS_IGNORE);
                unpack_matrix(tmp, buf);
                insert_col_block(full, tmp, b);
            }

            if (g_stats != nullptr)
            {
                g_stats->final_gather_ms += elapsed_ms(t_beg, MPI_Wtime());
            }
            return full;
        }

        std::vector<double> buf = pack_matrix(local_blk);
        if (g_stats != nullptr)
        {
            g_stats->logical_gather_bytes += static_cast<unsigned long long>(buf.size() * sizeof(double));
        }
        MPI_Send(buf.data(),
                 static_cast<int>(buf.size()),
                 MPI_DOUBLE,
                 root,
                 0,
                 comm);

        if (g_stats != nullptr)
        {
            g_stats->final_gather_ms += elapsed_ms(t_beg, MPI_Wtime());
        }
        return Matrix();
    }

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    // 即 M <- L * M，其中 L 只作用在第 r0/r1 两行上。
    // 这类逐元素线性组合很适合向量化，SIMD/多线程中你也可以顺手的事把他们做了。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        int col=M.cols();
        int j=0;
        double *row0=&M.at(r0,j);
        double *row1=&M.at(r1,j);
        float64x2_t vc=vdupq_n_f64(c);
        float64x2_t vs=vdupq_n_f64(s);
        float64x2_t vns=vdupq_n_f64(-s);
        //>八路
        for (; j+7<col;j+=8)
        {
            float64x2_t a0=vld1q_f64(row0+j);//load
            float64x2_t a1=vld1q_f64(row0+j+2);
            float64x2_t a2=vld1q_f64(row0+j+4);
            float64x2_t a3=vld1q_f64(row0+j+6);
            float64x2_t b0=vld1q_f64(row1+j);
            float64x2_t b1=vld1q_f64(row1+j+2);
            float64x2_t b2=vld1q_f64(row1+j+4);
            float64x2_t b3=vld1q_f64(row1+j+6);
            float64x2_t new00=vaddq_f64(vmulq_f64(vc,a0),vmulq_f64(vs,b0));//multi and sum
            float64x2_t new01=vaddq_f64(vmulq_f64(vns,a0),vmulq_f64(vc,b0));
            float64x2_t new10=vaddq_f64(vmulq_f64(vc,a1),vmulq_f64(vs,b1));
            float64x2_t new11=vaddq_f64(vmulq_f64(vns,a1),vmulq_f64(vc,b1));
            float64x2_t new20=vaddq_f64(vmulq_f64(vc,a2),vmulq_f64(vs,b2));
            float64x2_t new21=vaddq_f64(vmulq_f64(vns,a2),vmulq_f64(vc,b2));
            float64x2_t new30=vaddq_f64(vmulq_f64(vc,a3),vmulq_f64(vs,b3));
            float64x2_t new31=vaddq_f64(vmulq_f64(vns,a3),vmulq_f64(vc,b3));
            vst1q_f64(row0+j,new00);//store back
            vst1q_f64(row1+j,new01);
            vst1q_f64(row0+j+2,new10);
            vst1q_f64(row1+j+2,new11);
            vst1q_f64(row0+j+4,new20);
            vst1q_f64(row1+j+4,new21);
            vst1q_f64(row0+j+6,new30);
            vst1q_f64(row1+j+6,new31);
        }
        //>四路
        for (; j+3<col;j+=4)
        {
            float64x2_t a0=vld1q_f64(row0+j);//load
            float64x2_t a1=vld1q_f64(row0+j+2);
            float64x2_t b0=vld1q_f64(row1+j);
            float64x2_t b1=vld1q_f64(row1+j+2);
            float64x2_t new00=vaddq_f64(vmulq_f64(vc,a0),vmulq_f64(vs,b0));//multi and sum
            float64x2_t new01=vaddq_f64(vmulq_f64(vns,a0),vmulq_f64(vc,b0));
            float64x2_t new10=vaddq_f64(vmulq_f64(vc,a1),vmulq_f64(vs,b1));
            float64x2_t new11=vaddq_f64(vmulq_f64(vns,a1),vmulq_f64(vc,b1));
            vst1q_f64(row0+j,new00);//store back
            vst1q_f64(row1+j,new01);
            vst1q_f64(row0+j+2,new10);
            vst1q_f64(row1+j+2,new11);
        }
        //>二路
        for(;j+1<col;j+=2){
            float64x2_t a=vld1q_f64(row0+j);
            float64x2_t b=vld1q_f64(row1+j);
            float64x2_t new0=vaddq_f64(vmulq_f64(vc,a),vmulq_f64(vs,b));
            float64x2_t new1=vaddq_f64(vmulq_f64(vns,a),vmulq_f64(vc,b));
            vst1q_f64(row0+j,new0);
            vst1q_f64(row1+j,new1);
        }
        //>一路
        for(;j<col;j++){
            double a = M.at(r0, j);
            double b = M.at(r1, j);
            M.at(r0, j) = c * a + s * b;
            M.at(r1, j) = -s * a + c * b;
        }
    }

    // 对矩阵 M 的两列 c0, c1 右乘 Givens 旋转 [c s; -s c]。
    // 即 M <- M * R，其中 R 只作用在第 c0/c1 两列上。
    static void apply_right_cols_scalar(Matrix &M, int c0, int c1, double c, double s)
    {
        for (int i=0;i<M.rows();++i)
        {
            double a=M.at(i,c0);
            double b=M.at(i,c1);
            M.at(i,c0)=a*c-b*s;
            M.at(i,c1)=a*s+b*c;
        }
    }

    //对转置矩阵M^T的两行执行左乘，等价于原矩阵M的右乘两列。
    static void apply_right_cols(Matrix &Mt, int c0, int c1, double c, double s)
    {
        apply_left_rows(Mt,c0,c1,c,-s);
    }

    static void accumulate_left_into_U(Matrix &Ut, int r0, int r1, double c, double s)
    {
        // 我们该怎样积累 U 和 V 的更新呢？
        // 以此处 U 的积累为例，让我们B <- L * B 时，我们必须维护的等式是 A = U * B * V^T
        // 如果 A = U * B * V^T 不成立，那么我们最终的SVD结果显然不是 A 的正确分解。
        // 由于正交矩阵和其转置的乘积是I，一个自然的想法是让 U <- U * L^T。
        // 这样就变成 A = (U * L^T) * (L * B) * V^T = U * B * V^T，等式得以保持。

        // U <- U * L^T  <=>  U^T <- L * U^T，直接复用行连续访问的 SIMD 路径。
        apply_left_rows(Ut,r0,r1,c,s);
    }

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
    // 偏移用于加速 QR 迭代收敛，并让 bulge chasing 过程更稳定。
    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    // 将上二对角结构以外、且绝对值很小的元素强制置零。
    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        for (int i = 0; i < B.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                {
                    B.at(i, j) = 0.0;
                }
            }
        }
    }

    // 对活动块 [l, r] 执行一次“单块 GKH bulge chasing”迭代。
    // 流程：首次右乘引入 bulge -> 首次左乘消 bulge -> 交替右乘/左乘将 bulge 追赶到块末端。
    static void one_block_step_emit_ops(Matrix &B,int l,int r,std::vector<RowRotation> &u_ops,std::vector<RowRotation> &v_ops)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        // 首次右乘
        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols_scalar(B, l, l + 1, c, s);
        v_ops.push_back({l, l + 1, c, -s});

        // 首次左乘
        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows(B, l, l + 1, c, s);
        u_ops.push_back({l, l + 1, c, s});

        for (int k = l + 1; k <= r - 1; ++k)
        {
            // 右乘
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols_scalar(B, k, k + 1, c, s);
            v_ops.push_back({k, k + 1, c, -s});

            // 左乘
            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows(B, k, k + 1, c, s);
            u_ops.push_back({k, k + 1, c, s});
        }
    }

    // 处理“对角元 d_k 近零但超对角 e_k 未近零”的情况。
    // 思路与单块追赶类似：先右乘把 e_i 消掉，再左乘清理新引入的次对角 bulge，
    // 把这个问题逐步向右传递，直到块末端。
    static bool chase_zero_diagonal_emit_ops(Matrix &B,
                                         int k,
                                         double tol,
                                         std::vector<RowRotation> &u_ops,
                                         std::vector<RowRotation> &v_ops)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }

        // d_k ~ 0 且 e_k 还未收敛时，按 lim_1 思路进行压缩追赶：
        // 1) 右乘消去第 k 行的 e_k；2) 左乘消去引入的次对角 bulge；
        // 然后把问题传递到下一行，直到末端。
        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            // 右乘：使第 i 行满足 [d_i, e_i] * G = [r, 0]。
            givens_rotation(B.at(i,i),B.at(i,i+1),c,s,rr,false);
            apply_right_cols_scalar(B,i,i+1,c,s);
            v_ops.push_back({i, i + 1, c, -s});

            // 左乘：消去 (i+1, i) 处由右乘引入的 bulge。
            if (i + 1 < m)
            {
                givens_rotation(B.at(i,i),B.at(i+1,i),c,s,rr,true);
                apply_left_rows(B,i,i+1,c,s);
                u_ops.push_back({i, i + 1, c, s});
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    // 扫描所有 d_k≈0 的位置：若对应 e_k 仍显著非零，则调用追赶过程压缩该异常结构。
    // 返回值表示本轮是否对 B/U/V 做了实际更新。
    static bool handle_diagonal_zeros_emit_ops(Matrix &B,
                                           double tol,
                                           std::vector<RowRotation> &u_ops,
                                           std::vector<RowRotation> &v_ops)
    {
        const int n = B.cols();
        bool changed = false;

        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol &&
                std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal_emit_ops(B, k, tol, u_ops, v_ops))
                {
                    changed = true;
                }
            }
        }

        return changed;
    }

    // 根据超对角线是否“足够小”对问题进行分块。
    // 若 |e_k| <= tol*(|d_k|+|d_{k+1}|+1)，认为该位置可解耦并直接置零。
    // 最终会得到一系列小矩阵。
    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            int r = l;
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    static void append_rotations(std::vector<RowRotation> &dst, const std::vector<RowRotation> &src)
    {
        dst.insert(dst.end(), src.begin(), src.end());
    }

    static void shift_rotations(std::vector<RowRotation> &ops, int offset)
    {
        for (auto &op : ops)
        {
            op.r0 += offset;
            op.r1 += offset;
        }
    }

    static Matrix extract_square_block(const Matrix &B, int l, int r)
    {
        const int len = r - l + 1;
        Matrix blk(len, len, 0.0);
        for (int i = 0; i < len; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                blk.at(i, j) = B.at(l + i, l + j);
            }
        }
        return blk;
    }

    static void insert_square_block(Matrix &B, const Matrix &blk, int l)
    {
        for (int i = 0; i < blk.rows(); ++i)
        {
            for (int j = 0; j < blk.cols(); ++j)
            {
                B.at(l + i, l + j) = blk.at(i, j);
            }
        }
    }

    static void process_block_until_split(Matrix &localB,
                                          double tol,
                                          std::vector<RowRotation> &u_ops,
                                          std::vector<RowRotation> &v_ops,
                                          std::vector<Block> &child_blocks)
    {
        const int n = localB.cols();
        while (true)
        {
            std::vector<RowRotation> u_pre;
            std::vector<RowRotation> v_pre;

            cleanup_bidiagonal(localB, tol);
            handle_diagonal_zeros_emit_ops(localB, tol, u_pre, v_pre);
            append_rotations(u_ops, u_pre);
            append_rotations(v_ops, v_pre);

            child_blocks = split_active_blocks(localB, n, tol);

            int non_singleton_count = 0;
            for (const auto &blk : child_blocks)
            {
                if (blk.r > blk.l)
                {
                    ++non_singleton_count;
                }
            }

            if (non_singleton_count == 0 || child_blocks.size() > 1)
            {
                return;
            }

            std::vector<RowRotation> u_step;
            std::vector<RowRotation> v_step;
            one_block_step_emit_ops(localB, child_blocks[0].l, child_blocks[0].r, u_step, v_step);
            append_rotations(u_ops, u_step);
            append_rotations(v_ops, v_step);
        }
    }

    // 收尾步骤：
    // 1) 把奇异值（对角元）统一调整为非负；
    // 2) 按降序重排奇异值，同时同步重排 U、V 对应列。
    // 最终得到常见的 SVD 规范形式：sigma_1 >= sigma_2 >= ... >= 0。
    // 这个函数你不用太在意，后续任务也不会明确涉及它。
    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }
            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }
       

} // namespace

// 从“上二对角矩阵 B”出发执行 Golub-Kahan SVD 迭代（改进版）：
// - 输入输出满足 A = U * B * V^T 不变；
// - 迭代中自动分块、处理对角近零、并在每个活动块上做 bulge chasing；
// - 成功收敛后，B 被整理为非负且降序的对角矩阵（其对角元即奇异值）。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
    }
    
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    TimingStats stats;
    g_stats = &stats;
    const double t_total_beg = MPI_Wtime();

    bool converged = false;

    Matrix Ut_local;
    Matrix Vt_local;

    if (rank == 0)
    {
        Matrix Ut_full = transpose_copy(U);
        Matrix Vt_full = transpose_copy(V);
        Ut_local = scatter_col_blocks_from_root(Ut_full, m, m, 0, MPI_COMM_WORLD);
        Vt_local = scatter_col_blocks_from_root(Vt_full, n, n, 0, MPI_COMM_WORLD);
    }
    else
    {
        Matrix dummy_u;
        Matrix dummy_v;
        Ut_local = scatter_col_blocks_from_root(dummy_u, m, m, 0, MPI_COMM_WORLD);
        Vt_local = scatter_col_blocks_from_root(dummy_v, n, n, 0, MPI_COMM_WORLD);
    }

    int control = 0;
    std::deque<Block> task_pool;
    int task_budget = 0;

    if (rank == 0)
    {
        cleanup_bidiagonal(B, tol);
        std::vector<Block> init_blocks = split_active_blocks(B, n, tol);
        for (const auto &blk : init_blocks)
        {
            if (blk.r > blk.l)
            {
                task_pool.push_back(blk);
            }
        }
    }

    while (true)
    {
        std::vector<RowRotation> u_ops;
        std::vector<RowRotation> v_ops;

        if (rank == 0)
        {
            if (!task_pool.empty() && task_budget < max_iter)
            {
                Block task = task_pool.front();
                task_pool.pop_front();

                const double t_extract_beg = MPI_Wtime();
                Matrix localB = extract_square_block(B, task.l, task.r);
                const double t_extract_end = MPI_Wtime();
                stats.root_extract_insert_ms += elapsed_ms(t_extract_beg, t_extract_end);

                std::vector<Block> child_blocks;
                const double t_process_beg = MPI_Wtime();
                process_block_until_split(localB, tol, u_ops, v_ops, child_blocks);
                const double t_process_end = MPI_Wtime();
                stats.root_process_ms += elapsed_ms(t_process_beg, t_process_end);

                const double t_insert_beg = MPI_Wtime();
                insert_square_block(B, localB, task.l);
                shift_rotations(u_ops, task.l);
                shift_rotations(v_ops, task.l);
                const double t_insert_end = MPI_Wtime();
                stats.root_extract_insert_ms += elapsed_ms(t_insert_beg, t_insert_end);

                for (const auto &child : child_blocks)
                {
                    if (child.r > child.l)
                    {
                        task_pool.push_back({task.l + child.l, task.l + child.r});
                    }
                }

                control = 1;
                ++task_budget;
                stats.iterations += 1;
                stats.u_ops_total += static_cast<unsigned long long>(u_ops.size());
                stats.v_ops_total += static_cast<unsigned long long>(v_ops.size());
            }
            else
            {
                converged = task_pool.empty();
                control = 0;
            }
        }

        const double t_ctrl_beg = MPI_Wtime();
        MPI_Bcast(&control, 1, MPI_INT, 0, MPI_COMM_WORLD);
        const double t_ctrl_end = MPI_Wtime();
        stats.bcast_ctrl_ms += elapsed_ms(t_ctrl_beg, t_ctrl_end);
        if (rank == 0)
        {
            stats.logical_bcast_bytes += static_cast<unsigned long long>(sizeof(int)) *
                                         static_cast<unsigned long long>(size - 1);
        }
        if (control == 0)
        {
            break;
        }

        g_bcast_kind = 1;
        bcast_row_rotations(u_ops, 0, MPI_COMM_WORLD);
        g_bcast_kind = 2;
        bcast_row_rotations(v_ops, 0, MPI_COMM_WORLD);
        g_bcast_kind = 0;
        const double t_apply_ut_beg = MPI_Wtime();
        apply_row_rotations_local(Ut_local, u_ops);
        const double t_apply_ut_end = MPI_Wtime();
        stats.apply_ut_ms += elapsed_ms(t_apply_ut_beg, t_apply_ut_end);
        const double t_apply_vt_beg = MPI_Wtime();
        apply_row_rotations_local(Vt_local, v_ops);
        const double t_apply_vt_end = MPI_Wtime();
        stats.apply_vt_ms += elapsed_ms(t_apply_vt_beg, t_apply_vt_end);
    }

    Matrix Ut_gathered = gather_col_blocks_to_root(Ut_local, m, m, 0, MPI_COMM_WORLD);
    Matrix Vt_gathered = gather_col_blocks_to_root(Vt_local, n, n, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        const double t_finalize_beg = MPI_Wtime();
        U = transpose_copy(Ut_gathered);
        V = transpose_copy(Vt_gathered);

        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
        {
            B.at(i, i + 1) = 0.0;
        }
        make_nonnegative_and_sort(U, B, V);
        const double t_finalize_end = MPI_Wtime();
        stats.root_finalize_ms += elapsed_ms(t_finalize_beg, t_finalize_end);
    }

    stats.total_wall_ms = elapsed_ms(t_total_beg, MPI_Wtime());
    print_timing_summary_mpi2(stats, rank);
    g_stats = nullptr;
    int conv_int = converged ? 1 : 0;
    MPI_Bcast(&conv_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return conv_int != 0;
}
