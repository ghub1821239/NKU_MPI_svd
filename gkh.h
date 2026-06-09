#pragma once

#include "matrix.h"

#include <cstdint>

struct GkhRunStats
{
    int mpi_size = 1;

    double total_wall_ms = 0.0;

    double task_prepare_ms_root = 0.0;
    double task_send_ms_root = 0.0;

    double task_wait_recv_ms_sum = 0.0;
    double task_wait_recv_ms_max = 0.0;
    double task_unpack_ms_sum = 0.0;
    double task_unpack_ms_max = 0.0;

    double compute_ms_sum = 0.0;
    double compute_ms_max = 0.0;

    double result_pack_ms_sum = 0.0;
    double result_pack_ms_max = 0.0;
    double result_send_ms_sum = 0.0;
    double result_send_ms_max = 0.0;

    double result_wait_recv_ms_root = 0.0;
    double result_unpack_ms_root = 0.0;
    double merge_ms_root = 0.0;
    double finalize_ms_root = 0.0;
    double stop_send_ms_root = 0.0;

    std::uint64_t tasks_dispatched = 0;
    std::uint64_t tasks_executed = 0;
    std::uint64_t task_bytes_sent = 0;
    std::uint64_t result_bytes_sent = 0;
};

// 在已上二对角化结果 A = U * B * V^T 上执行 GKH 迭代。
// 额外处理“主对角线收敛到 0”的压缩情形。
//
// 输入要求：
// - B 为 m x n 且 m >= n，且近似上二对角
// - U 为 m x m，V 为 n x n
//
// 输出：
// - 保持 A = U * B * V^T
// - 若收敛，B 变为非负降序对角（m x n）
//
// 返回：是否在 max_iter 轮内收敛。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V,
                             int max_iter = 6000,
                             double tol = 1e-12);

const GkhRunStats &gkh_last_run_stats();
