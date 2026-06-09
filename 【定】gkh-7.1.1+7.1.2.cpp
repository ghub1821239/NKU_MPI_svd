#include "gkh.h"
#include "givens.h"
#include <arm_neon.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

    static constexpr int TAG_TASK_HDR = 100;
    static constexpr int TAG_TASK_DATA = 101;
    static constexpr int TAG_RESULT_META = 200;
    static constexpr int TAG_RESULT_DATA = 201;
    static constexpr int TAG_RESULT_BLOCKS = 202;
    static constexpr int TAG_RESULT_UOPS = 203;
    static constexpr int TAG_RESULT_VOPS = 204;

    struct TimingStats
    {
        double task_prepare_ms = 0.0;
        double task_send_ms = 0.0;
        double task_wait_recv_ms = 0.0;
        double task_unpack_ms = 0.0;
        double compute_ms = 0.0;
        double result_pack_ms = 0.0;
        double result_send_ms = 0.0;
        double result_wait_recv_ms = 0.0;
        double result_unpack_ms = 0.0;
        double merge_apply_ms = 0.0;
        double finalize_ms = 0.0;
        double stop_send_ms = 0.0;
        double total_wall_ms = 0.0;

        unsigned long long tasks_dispatched = 0;
        unsigned long long tasks_executed = 0;
        unsigned long long task_bytes_sent = 0;
        unsigned long long result_bytes_sent = 0;
    };

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

    static void print_timing_summary_7_1(const TimingStats &local, int rank)
    {
        double task_prepare_sum = 0.0, task_prepare_max = 0.0;
        double task_send_sum = 0.0, task_send_max = 0.0;
        double task_wait_recv_sum = 0.0, task_wait_recv_max = 0.0;
        double task_unpack_sum = 0.0, task_unpack_max = 0.0;
        double compute_sum = 0.0, compute_max = 0.0;
        double result_pack_sum = 0.0, result_pack_max = 0.0;
        double result_send_sum = 0.0, result_send_max = 0.0;
        double result_wait_recv_sum = 0.0, result_wait_recv_max = 0.0;
        double result_unpack_sum = 0.0, result_unpack_max = 0.0;
        double merge_apply_sum = 0.0, merge_apply_max = 0.0;
        double finalize_sum = 0.0, finalize_max = 0.0;
        double stop_send_sum = 0.0, stop_send_max = 0.0;
        double total_wall_max = 0.0;

        reduce_sum_max(local.task_prepare_ms, task_prepare_sum, task_prepare_max);
        reduce_sum_max(local.task_send_ms, task_send_sum, task_send_max);
        reduce_sum_max(local.task_wait_recv_ms, task_wait_recv_sum, task_wait_recv_max);
        reduce_sum_max(local.task_unpack_ms, task_unpack_sum, task_unpack_max);
        reduce_sum_max(local.compute_ms, compute_sum, compute_max);
        reduce_sum_max(local.result_pack_ms, result_pack_sum, result_pack_max);
        reduce_sum_max(local.result_send_ms, result_send_sum, result_send_max);
        reduce_sum_max(local.result_wait_recv_ms, result_wait_recv_sum, result_wait_recv_max);
        reduce_sum_max(local.result_unpack_ms, result_unpack_sum, result_unpack_max);
        reduce_sum_max(local.merge_apply_ms, merge_apply_sum, merge_apply_max);
        reduce_sum_max(local.finalize_ms, finalize_sum, finalize_max);
        reduce_sum_max(local.stop_send_ms, stop_send_sum, stop_send_max);
        MPI_Reduce(&local.total_wall_ms, &total_wall_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        unsigned long long tasks_dispatched = 0;
        unsigned long long tasks_executed = 0;
        unsigned long long task_bytes_sent = 0;
        unsigned long long result_bytes_sent = 0;
        reduce_sum_ull(local.tasks_dispatched, tasks_dispatched);
        reduce_sum_ull(local.tasks_executed, tasks_executed);
        reduce_sum_ull(local.task_bytes_sent, task_bytes_sent);
        reduce_sum_ull(local.result_bytes_sent, result_bytes_sent);

        if (rank != 0)
        {
            return;
        }

        const double comm_sum = task_send_sum + task_wait_recv_sum + result_send_sum +
                                result_wait_recv_sum + stop_send_sum;
        const double serdes_sum = task_prepare_sum + task_unpack_sum + result_pack_sum +
                                  result_unpack_sum;

        std::cout << "[7.1 master-worker timing]\n";
        std::cout << "  total wall(ms, MPI_MAX)   : " << total_wall_max << "\n";
        std::cout << "  task prepare(ms)          : sum " << task_prepare_sum << ", max " << task_prepare_max << "\n";
        std::cout << "  task send(ms)             : sum " << task_send_sum << ", max " << task_send_max << "\n";
        std::cout << "  task wait recv(ms)        : sum " << task_wait_recv_sum << ", max " << task_wait_recv_max << "\n";
        std::cout << "  task unpack(ms)           : sum " << task_unpack_sum << ", max " << task_unpack_max << "\n";
        std::cout << "  compute(ms)               : sum " << compute_sum << ", max " << compute_max << "\n";
        std::cout << "  result pack(ms)           : sum " << result_pack_sum << ", max " << result_pack_max << "\n";
        std::cout << "  result send(ms)           : sum " << result_send_sum << ", max " << result_send_max << "\n";
        std::cout << "  result wait recv(ms)      : sum " << result_wait_recv_sum << ", max " << result_wait_recv_max << "\n";
        std::cout << "  result unpack(ms)         : sum " << result_unpack_sum << ", max " << result_unpack_max << "\n";
        std::cout << "  merge+apply(ms)           : sum " << merge_apply_sum << ", max " << merge_apply_max << "\n";
        std::cout << "  finalize(ms)             : sum " << finalize_sum << ", max " << finalize_max << "\n";
        std::cout << "  stop send(ms)             : sum " << stop_send_sum << ", max " << stop_send_max << "\n";
        std::cout << "  comm subtotal(ms, sum)    : " << comm_sum << "\n";
        std::cout << "  serdes subtotal(ms, sum)  : " << serdes_sum << "\n";
        std::cout << "  tasks dispatched/executed : " << tasks_dispatched << " / " << tasks_executed << "\n";
        std::cout << "  payload bytes task/result : " << task_bytes_sent << " / " << result_bytes_sent << "\n";
    }

    static Matrix transpose_copy(const Matrix &A)
    {
        Matrix T(A.cols(), A.rows(), 0.0);
        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < A.cols(); ++j)
            {
                T.at(j, i) = A.at(i, j);
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

    struct RowRotation
    {
        int r0;
        int r1;
        double c;
        double s;
    };

    struct TaskResult
    {
        int source;
        Block parent;
        Matrix block_matrix;
        std::vector<Block> child_blocks;
        std::vector<RowRotation> u_ops;
        std::vector<RowRotation> v_ops;
    };

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    // 即 M <- L * M，其中 L 只作用在第 r0/r1 两行上。
    // 这类逐元素线性组合很适合向量化，SIMD/多线程中你也可以顺手的事把他们做了。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        int col = M.cols();
        int j = 0;
        double *row0 = &M.at(r0, j);
        double *row1 = &M.at(r1, j);
        float64x2_t vc = vdupq_n_f64(c);
        float64x2_t vs = vdupq_n_f64(s);
        float64x2_t vns = vdupq_n_f64(-s);

        for (; j + 7 < col; j += 8)
        {
            float64x2_t a0 = vld1q_f64(row0 + j);
            float64x2_t a1 = vld1q_f64(row0 + j + 2);
            float64x2_t a2 = vld1q_f64(row0 + j + 4);
            float64x2_t a3 = vld1q_f64(row0 + j + 6);
            float64x2_t b0 = vld1q_f64(row1 + j);
            float64x2_t b1 = vld1q_f64(row1 + j + 2);
            float64x2_t b2 = vld1q_f64(row1 + j + 4);
            float64x2_t b3 = vld1q_f64(row1 + j + 6);
            float64x2_t new00 = vaddq_f64(vmulq_f64(vc, a0), vmulq_f64(vs, b0));
            float64x2_t new01 = vaddq_f64(vmulq_f64(vns, a0), vmulq_f64(vc, b0));
            float64x2_t new10 = vaddq_f64(vmulq_f64(vc, a1), vmulq_f64(vs, b1));
            float64x2_t new11 = vaddq_f64(vmulq_f64(vns, a1), vmulq_f64(vc, b1));
            float64x2_t new20 = vaddq_f64(vmulq_f64(vc, a2), vmulq_f64(vs, b2));
            float64x2_t new21 = vaddq_f64(vmulq_f64(vns, a2), vmulq_f64(vc, b2));
            float64x2_t new30 = vaddq_f64(vmulq_f64(vc, a3), vmulq_f64(vs, b3));
            float64x2_t new31 = vaddq_f64(vmulq_f64(vns, a3), vmulq_f64(vc, b3));
            vst1q_f64(row0 + j, new00);
            vst1q_f64(row1 + j, new01);
            vst1q_f64(row0 + j + 2, new10);
            vst1q_f64(row1 + j + 2, new11);
            vst1q_f64(row0 + j + 4, new20);
            vst1q_f64(row1 + j + 4, new21);
            vst1q_f64(row0 + j + 6, new30);
            vst1q_f64(row1 + j + 6, new31);
        }

        for (; j + 3 < col; j += 4)
        {
            float64x2_t a0 = vld1q_f64(row0 + j);
            float64x2_t a1 = vld1q_f64(row0 + j + 2);
            float64x2_t b0 = vld1q_f64(row1 + j);
            float64x2_t b1 = vld1q_f64(row1 + j + 2);
            float64x2_t new00 = vaddq_f64(vmulq_f64(vc, a0), vmulq_f64(vs, b0));
            float64x2_t new01 = vaddq_f64(vmulq_f64(vns, a0), vmulq_f64(vc, b0));
            float64x2_t new10 = vaddq_f64(vmulq_f64(vc, a1), vmulq_f64(vs, b1));
            float64x2_t new11 = vaddq_f64(vmulq_f64(vns, a1), vmulq_f64(vc, b1));
            vst1q_f64(row0 + j, new00);
            vst1q_f64(row1 + j, new01);
            vst1q_f64(row0 + j + 2, new10);
            vst1q_f64(row1 + j + 2, new11);
        }

        for (; j + 1 < col; j += 2)
        {
            float64x2_t a = vld1q_f64(row0 + j);
            float64x2_t b = vld1q_f64(row1 + j);
            float64x2_t new0 = vaddq_f64(vmulq_f64(vc, a), vmulq_f64(vs, b));
            float64x2_t new1 = vaddq_f64(vmulq_f64(vns, a), vmulq_f64(vc, b));
            vst1q_f64(row0 + j, new0);
            vst1q_f64(row1 + j, new1);
        }

        for (; j < col; ++j)
        {
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
        for (int i = 0; i < M.rows(); ++i)
        {
            double a = M.at(i, c0);
            double b = M.at(i, c1);
            M.at(i, c0) = a * c - b * s;
            M.at(i, c1) = a * s + b * c;
        }
    }

    // 对转置矩阵 M^T 的两行执行左乘，等价于原矩阵 M 的右乘两列。
    static void apply_right_cols(Matrix &Mt, int c0, int c1, double c, double s)
    {
        apply_left_rows(Mt, c0, c1, c, -s);
    }

    static void accumulate_left_into_U(Matrix &Ut, int r0, int r1, double c, double s)
    {
        // 我们该怎样积累 U 和 V 的更新呢？
        // 以此处 U 的积累为例，让我们 B <- L * B 时，我们必须维护的等式是 A = U * B * V^T。
        // 由于 U <- U * L^T  <=>  U^T <- L * U^T，可以直接复用行连续访问的 SIMD 路径。
        apply_left_rows(Ut, r0, r1, c, s);
    }

    static void apply_row_rotations_local(Matrix &M, const std::vector<RowRotation> &ops)
    {
        for (const auto &op : ops)
        {
            apply_left_rows(M, op.r0, op.r1, op.c, op.s);
        }
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

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
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

    // 根据超对角线是否“足够小”对问题进行分块。
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

    static void one_block_step_emit_ops(Matrix &B, int l, int r,
                                        std::vector<RowRotation> &u_ops,
                                        std::vector<RowRotation> &v_ops)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);
        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols_scalar(B, l, l + 1, c, s);
        v_ops.push_back({l, l + 1, c, -s});

        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows(B, l, l + 1, c, s);
        u_ops.push_back({l, l + 1, c, s});

        for (int k = l + 1; k <= r - 1; ++k)
        {
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols_scalar(B, k, k + 1, c, s);
            v_ops.push_back({k, k + 1, c, -s});

            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows(B, k, k + 1, c, s);
            u_ops.push_back({k, k + 1, c, s});
        }
    }

    static bool chase_zero_diagonal_emit_ops(Matrix &B, int k, double tol,
                                             std::vector<RowRotation> &u_ops,
                                             std::vector<RowRotation> &v_ops)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }

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

            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
            apply_right_cols_scalar(B, i, i + 1, c, s);
            v_ops.push_back({i, i + 1, c, -s});

            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
                apply_left_rows(B, i, i + 1, c, s);
                u_ops.push_back({i, i + 1, c, s});
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    static bool handle_diagonal_zeros_emit_ops(Matrix &B, double tol,
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
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal_emit_ops(B, k, tol, u_ops, v_ops))
                {
                    changed = true;
                }
            }
        }

        return changed;
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

    static std::vector<int> pack_blocks(const std::vector<Block> &blocks)
    {
        std::vector<int> data;
        data.reserve(blocks.size() * 2);
        for (const auto &blk : blocks)
        {
            data.push_back(blk.l);
            data.push_back(blk.r);
        }
        return data;
    }

    static std::vector<Block> unpack_blocks(const std::vector<int> &data)
    {
        std::vector<Block> blocks;
        for (size_t i = 0; i + 1 < data.size(); i += 2)
        {
            blocks.push_back({data[i], data[i + 1]});
        }
        return blocks;
    }

    static void process_block_until_split(Matrix &localB, double tol,
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

    static void send_task_to_worker(const Matrix &B, const Block &blk, int worker, MPI_Comm comm, TimingStats &stats)
    {
        const double t_prepare_beg = MPI_Wtime();
        Matrix localB = extract_square_block(B, blk.l, blk.r);
        std::vector<double> buf = pack_matrix(localB);
        const double t_prepare_end = MPI_Wtime();
        stats.task_prepare_ms += elapsed_ms(t_prepare_beg, t_prepare_end);

        const int hdr[3] = {1, blk.l, blk.r};
        const double t_send_beg = MPI_Wtime();
        MPI_Send(const_cast<int *>(hdr), 3, MPI_INT, worker, TAG_TASK_HDR, comm);
        if (!buf.empty())
        {
            MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, worker, TAG_TASK_DATA, comm);
        }
        const double t_send_end = MPI_Wtime();
        stats.task_send_ms += elapsed_ms(t_send_beg, t_send_end);
        stats.tasks_dispatched += 1;
        stats.task_bytes_sent += static_cast<unsigned long long>(sizeof(hdr));
        stats.task_bytes_sent += static_cast<unsigned long long>(buf.size() * sizeof(double));
    }

    static void send_stop_to_worker(int worker, MPI_Comm comm, TimingStats &stats)
    {
        const int hdr[3] = {0, 0, 0};
        const double t_send_beg = MPI_Wtime();
        MPI_Send(const_cast<int *>(hdr), 3, MPI_INT, worker, TAG_TASK_HDR, comm);
        const double t_send_end = MPI_Wtime();
        stats.stop_send_ms += elapsed_ms(t_send_beg, t_send_end);
    }

    static bool recv_task_from_root(Block &blk, Matrix &localB, MPI_Comm comm, TimingStats &stats)
    {
        int hdr[3];
        const double t_wait_hdr_beg = MPI_Wtime();
        MPI_Recv(hdr, 3, MPI_INT, 0, TAG_TASK_HDR, comm, MPI_STATUS_IGNORE);
        const double t_wait_hdr_end = MPI_Wtime();
        stats.task_wait_recv_ms += elapsed_ms(t_wait_hdr_beg, t_wait_hdr_end);
        if (hdr[0] == 0)
        {
            return false;
        }

        blk = {hdr[1], hdr[2]};
        const int len = blk.r - blk.l + 1;
        localB = Matrix(len, len, 0.0);
        std::vector<double> buf(static_cast<size_t>(len) * static_cast<size_t>(len));
        if (!buf.empty())
        {
            const double t_wait_data_beg = MPI_Wtime();
            MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, 0, TAG_TASK_DATA, comm, MPI_STATUS_IGNORE);
            const double t_wait_data_end = MPI_Wtime();
            stats.task_wait_recv_ms += elapsed_ms(t_wait_data_beg, t_wait_data_end);
            const double t_unpack_beg = MPI_Wtime();
            unpack_matrix(localB, buf);
            const double t_unpack_end = MPI_Wtime();
            stats.task_unpack_ms += elapsed_ms(t_unpack_beg, t_unpack_end);
        }
        return true;
    }

    static void send_result_to_root(const Block &parent, const Matrix &localB,
                                    const std::vector<Block> &child_blocks,
                                    const std::vector<RowRotation> &u_ops,
                                    const std::vector<RowRotation> &v_ops,
                                    MPI_Comm comm,
                                    TimingStats &stats)
    {
        const double t_pack_beg = MPI_Wtime();
        const int meta[5] = {
            parent.l,
            parent.r,
            static_cast<int>(child_blocks.size()),
            static_cast<int>(u_ops.size()),
            static_cast<int>(v_ops.size())
        };
        std::vector<double> block_buf = pack_matrix(localB);
        std::vector<int> block_meta = pack_blocks(child_blocks);
        const double t_pack_end = MPI_Wtime();
        stats.result_pack_ms += elapsed_ms(t_pack_beg, t_pack_end);

        const double t_send_beg = MPI_Wtime();
        MPI_Send(const_cast<int *>(meta), 5, MPI_INT, 0, TAG_RESULT_META, comm);

        if (!block_buf.empty())
        {
            MPI_Send(block_buf.data(), static_cast<int>(block_buf.size()), MPI_DOUBLE, 0, TAG_RESULT_DATA, comm);
        }

        if (!block_meta.empty())
        {
            MPI_Send(block_meta.data(), static_cast<int>(block_meta.size()), MPI_INT, 0, TAG_RESULT_BLOCKS, comm);
        }

        if (!u_ops.empty())
        {
            MPI_Send(reinterpret_cast<const void *>(u_ops.data()),
                     static_cast<int>(u_ops.size() * sizeof(RowRotation)),
                     MPI_BYTE,
                     0,
                     TAG_RESULT_UOPS,
                     comm);
        }

        if (!v_ops.empty())
        {
            MPI_Send(reinterpret_cast<const void *>(v_ops.data()),
                     static_cast<int>(v_ops.size() * sizeof(RowRotation)),
                     MPI_BYTE,
                     0,
                     TAG_RESULT_VOPS,
                     comm);
        }
        const double t_send_end = MPI_Wtime();
        stats.result_send_ms += elapsed_ms(t_send_beg, t_send_end);
        stats.result_bytes_sent += static_cast<unsigned long long>(sizeof(meta));
        stats.result_bytes_sent += static_cast<unsigned long long>(block_buf.size() * sizeof(double));
        stats.result_bytes_sent += static_cast<unsigned long long>(block_meta.size() * sizeof(int));
        stats.result_bytes_sent += static_cast<unsigned long long>(u_ops.size() * sizeof(RowRotation));
        stats.result_bytes_sent += static_cast<unsigned long long>(v_ops.size() * sizeof(RowRotation));
    }

    static TaskResult recv_result_from_any_worker(MPI_Comm comm, TimingStats &stats)
    {
        MPI_Status status;
        int meta[5];
        const double t_wait_meta_beg = MPI_Wtime();
        MPI_Recv(meta, 5, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT_META, comm, &status);
        const double t_wait_meta_end = MPI_Wtime();
        stats.result_wait_recv_ms += elapsed_ms(t_wait_meta_beg, t_wait_meta_end);

        TaskResult result;
        result.source = status.MPI_SOURCE;
        result.parent = {meta[0], meta[1]};

        const int len = result.parent.r - result.parent.l + 1;
        result.block_matrix = Matrix(len, len, 0.0);
        std::vector<double> block_buf(static_cast<size_t>(len) * static_cast<size_t>(len));
        if (!block_buf.empty())
        {
            const double t_wait_block_beg = MPI_Wtime();
            MPI_Recv(block_buf.data(),
                     static_cast<int>(block_buf.size()),
                     MPI_DOUBLE,
                     result.source,
                     TAG_RESULT_DATA,
                     comm,
                     MPI_STATUS_IGNORE);
            const double t_wait_block_end = MPI_Wtime();
            stats.result_wait_recv_ms += elapsed_ms(t_wait_block_beg, t_wait_block_end);
            const double t_unpack_block_beg = MPI_Wtime();
            unpack_matrix(result.block_matrix, block_buf);
            const double t_unpack_block_end = MPI_Wtime();
            stats.result_unpack_ms += elapsed_ms(t_unpack_block_beg, t_unpack_block_end);
        }

        if (meta[2] > 0)
        {
            std::vector<int> child_buf(static_cast<size_t>(meta[2]) * 2U);
            const double t_wait_child_beg = MPI_Wtime();
            MPI_Recv(child_buf.data(),
                     static_cast<int>(child_buf.size()),
                     MPI_INT,
                     result.source,
                     TAG_RESULT_BLOCKS,
                     comm,
                     MPI_STATUS_IGNORE);
            const double t_wait_child_end = MPI_Wtime();
            stats.result_wait_recv_ms += elapsed_ms(t_wait_child_beg, t_wait_child_end);
            const double t_unpack_child_beg = MPI_Wtime();
            result.child_blocks = unpack_blocks(child_buf);
            const double t_unpack_child_end = MPI_Wtime();
            stats.result_unpack_ms += elapsed_ms(t_unpack_child_beg, t_unpack_child_end);
        }

        if (meta[3] > 0)
        {
            result.u_ops.resize(meta[3]);
            const double t_wait_uops_beg = MPI_Wtime();
            MPI_Recv(reinterpret_cast<void *>(result.u_ops.data()),
                     static_cast<int>(meta[3] * sizeof(RowRotation)),
                     MPI_BYTE,
                     result.source,
                     TAG_RESULT_UOPS,
                     comm,
                     MPI_STATUS_IGNORE);
            const double t_wait_uops_end = MPI_Wtime();
            stats.result_wait_recv_ms += elapsed_ms(t_wait_uops_beg, t_wait_uops_end);
        }

        if (meta[4] > 0)
        {
            result.v_ops.resize(meta[4]);
            const double t_wait_vops_beg = MPI_Wtime();
            MPI_Recv(reinterpret_cast<void *>(result.v_ops.data()),
                     static_cast<int>(meta[4] * sizeof(RowRotation)),
                     MPI_BYTE,
                     result.source,
                     TAG_RESULT_VOPS,
                     comm,
                     MPI_STATUS_IGNORE);
            const double t_wait_vops_end = MPI_Wtime();
            stats.result_wait_recv_ms += elapsed_ms(t_wait_vops_beg, t_wait_vops_end);
        }

        return result;
    }

    // 收尾步骤：
    // 1) 把奇异值（对角元）统一调整为非负；
    // 2) 按降序重排奇异值，同时同步重排 U、V 对应列。
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
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return B.at(a, a) > B.at(b, b); });

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
// - 7.1.2：不再每轮扫描所有块，而是维护一个非 1x1 块任务池；
// - 7.1.1：rank 0 管理任务池，worker 进程领取任务并回传结果。
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    TimingStats stats;
    const double t_total_beg = MPI_Wtime();

    Matrix Ut;
    Matrix Vt;
    if (rank == 0)
    {
        Ut = transpose_copy(U);
        Vt = transpose_copy(V);
    }

    if (size == 1)
    {
        std::deque<Block> task_pool;
        cleanup_bidiagonal(B, tol);
        std::vector<Block> init_blocks = split_active_blocks(B, n, tol);
        for (const auto &blk : init_blocks)
        {
            if (blk.r > blk.l)
            {
                task_pool.push_back(blk);
            }
        }

        int task_budget = 0;
        while (!task_pool.empty() && task_budget < max_iter)
        {
            Block task = task_pool.front();
            task_pool.pop_front();

            const double t_prepare_beg = MPI_Wtime();
            Matrix localB = extract_square_block(B, task.l, task.r);
            const double t_prepare_end = MPI_Wtime();
            stats.task_prepare_ms += elapsed_ms(t_prepare_beg, t_prepare_end);
            std::vector<RowRotation> u_ops;
            std::vector<RowRotation> v_ops;
            std::vector<Block> child_blocks;
            const double t_compute_beg = MPI_Wtime();
            process_block_until_split(localB, tol, u_ops, v_ops, child_blocks);
            const double t_compute_end = MPI_Wtime();
            stats.compute_ms += elapsed_ms(t_compute_beg, t_compute_end);
            stats.tasks_executed += 1;

            const double t_merge_beg = MPI_Wtime();
            insert_square_block(B, localB, task.l);
            shift_rotations(u_ops, task.l);
            shift_rotations(v_ops, task.l);
            apply_row_rotations_local(Ut, u_ops);
            apply_row_rotations_local(Vt, v_ops);
            const double t_merge_end = MPI_Wtime();
            stats.merge_apply_ms += elapsed_ms(t_merge_beg, t_merge_end);

            for (const auto &child : child_blocks)
            {
                if (child.r > child.l)
                {
                    task_pool.push_back({task.l + child.l, task.l + child.r});
                }
            }

            ++task_budget;
        }

        const double t_finalize_beg = MPI_Wtime();
        U = transpose_copy(Ut);
        V = transpose_copy(Vt);
        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
        {
            B.at(i, i + 1) = 0.0;
        }
        make_nonnegative_and_sort(U, B, V);
        const double t_finalize_end = MPI_Wtime();
        stats.finalize_ms += elapsed_ms(t_finalize_beg, t_finalize_end);
        stats.total_wall_ms = elapsed_ms(t_total_beg, MPI_Wtime());
        print_timing_summary_7_1(stats, rank);
        return task_pool.empty();
    }

    if (rank == 0)
    {
        std::deque<Block> task_pool;
        cleanup_bidiagonal(B, tol);
        std::vector<Block> init_blocks = split_active_blocks(B, n, tol);
        for (const auto &blk : init_blocks)
        {
            if (blk.r > blk.l)
            {
                task_pool.push_back(blk);
            }
        }

        int active_workers = 0;
        int task_budget = 0;

        for (int worker = 1; worker < size && !task_pool.empty() && task_budget < max_iter; ++worker)
        {
            send_task_to_worker(B, task_pool.front(), worker, MPI_COMM_WORLD, stats);
            task_pool.pop_front();
            ++active_workers;
            ++task_budget;
        }

        while (active_workers > 0)
        {
            TaskResult result = recv_result_from_any_worker(MPI_COMM_WORLD, stats);
            --active_workers;

            const double t_merge_beg = MPI_Wtime();
            insert_square_block(B, result.block_matrix, result.parent.l);
            shift_rotations(result.u_ops, result.parent.l);
            shift_rotations(result.v_ops, result.parent.l);
            apply_row_rotations_local(Ut, result.u_ops);
            apply_row_rotations_local(Vt, result.v_ops);
            const double t_merge_end = MPI_Wtime();
            stats.merge_apply_ms += elapsed_ms(t_merge_beg, t_merge_end);

            for (const auto &child : result.child_blocks)
            {
                if (child.r > child.l)
                {
                    task_pool.push_back({result.parent.l + child.l, result.parent.l + child.r});
                }
            }

            if (!task_pool.empty() && task_budget < max_iter)
            {
                send_task_to_worker(B, task_pool.front(), result.source, MPI_COMM_WORLD, stats);
                task_pool.pop_front();
                ++active_workers;
                ++task_budget;
            }
        }

        for (int worker = 1; worker < size; ++worker)
        {
            send_stop_to_worker(worker, MPI_COMM_WORLD, stats);
        }

        const double t_finalize_beg = MPI_Wtime();
        U = transpose_copy(Ut);
        V = transpose_copy(Vt);
        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
        {
            B.at(i, i + 1) = 0.0;
        }
        make_nonnegative_and_sort(U, B, V);
        const double t_finalize_end = MPI_Wtime();
        stats.finalize_ms += elapsed_ms(t_finalize_beg, t_finalize_end);
        stats.total_wall_ms = elapsed_ms(t_total_beg, MPI_Wtime());
        print_timing_summary_7_1(stats, rank);
        return task_pool.empty();
    }

        while (true)
        {
            Block task;
            Matrix localB;
        if (!recv_task_from_root(task, localB, MPI_COMM_WORLD, stats))
        {
            break;
        }

        std::vector<RowRotation> u_ops;
        std::vector<RowRotation> v_ops;
        std::vector<Block> child_blocks;
        const double t_compute_beg = MPI_Wtime();
        process_block_until_split(localB, tol, u_ops, v_ops, child_blocks);
        const double t_compute_end = MPI_Wtime();
        stats.compute_ms += elapsed_ms(t_compute_beg, t_compute_end);
        stats.tasks_executed += 1;
        send_result_to_root(task, localB, child_blocks, u_ops, v_ops, MPI_COMM_WORLD, stats);
    }

    stats.total_wall_ms = elapsed_ms(t_total_beg, MPI_Wtime());
    print_timing_summary_7_1(stats, rank);
    return false;
}
