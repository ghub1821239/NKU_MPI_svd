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

    bool converged = false;

    Matrix Ut_local;
    Matrix Vt_local;

    if (rank == 0)
    {
        // 只在 rank 0 上保留完整 U/V，然后转置成更适合行更新的 Ut/Vt。
        // 接着直接把各 rank 真正需要的列块发出去，避免一开始把完整矩阵复制到所有进程。
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

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // 本轮由 rank 0 生成的 U/V 旋转操作。
        std::vector<RowRotation> u_ops;
        std::vector<RowRotation> v_ops;

        // rank 0 上做活动块拆分和 B 主链推进；
        // 其他进程只等待广播来的旋转参数，在本地更新自己的 Ut/Vt 列块。
        std::vector<Block> blocks;
        int all_singletons = 0;
        int task_count = 0;

        if (rank == 0)
        {
            // 清理数值噪声，并优先处理 d_k≈0 的特殊情形。
            cleanup_bidiagonal(B, tol);
            handle_diagonal_zeros_emit_ops(B, tol, u_ops, v_ops);

            // 根据超对角线断点拆分活动块
            // 这里子矩阵间是相互独立的，所以此处具有很大的并行潜力：你可以尝试多线程/多进程进行处理
            // 但根据算法，收集 Givens 旋转并更新 U/V 需要在每个块内顺序执行，所以这可能给并行带来麻烦。
            std::vector<Block> tmp_blocks = split_active_blocks(B, n, tol);
            blocks.swap(tmp_blocks);

            /*[log]
            int non_singleton_count = 0;

            std::cerr << "[gkh] iter=" << iter
                    << " total_blocks=" << blocks.size()
                    << " lens=[";

            for (size_t bi = 0; bi < blocks.size(); ++bi) {
                int len = blocks[bi].r - blocks[bi].l + 1;
                if (len > 1) {
                    ++non_singleton_count;
                }
                if (bi > 0) {
                    std::cerr << ",";
                }
                std::cerr << len;
            }

            std::cerr << "] non_singleton=" << non_singleton_count << "\n";
            */

            // 若全部是 1x1 块，说明所有超对角都已收敛为 0。
            all_singletons = 1;
            for (const auto &blk : blocks)
            {
                if (blk.r > blk.l)
                {
                    all_singletons = 0;
                    ++task_count;
                }
            }

            if (all_singletons)
            {
                converged = true;
            }
        }

        // 先广播由 handle_diagonal_zeros 阶段产生的 U/V 更新
        bcast_row_rotations(u_ops, 0, MPI_COMM_WORLD);
        bcast_row_rotations(v_ops, 0, MPI_COMM_WORLD);

        apply_row_rotations_local(Ut_local, u_ops);
        apply_row_rotations_local(Vt_local, v_ops);

        MPI_Bcast(&all_singletons, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (all_singletons)
        {
            break;
        }

        MPI_Bcast(&task_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 从右到左处理每个非平凡块，减少末端块对前面块的干扰。
        // rank 0 串行推进 B，并把每个块对应的 U/V 更新操作广播出去；
        // 所有进程在本地列块上应用这些操作。
        int idx = static_cast<int>(blocks.size()) - 1;
        for (int t = 0; t < task_count; ++t)
        {
            std::vector<RowRotation> u_step_ops;
            std::vector<RowRotation> v_step_ops;

            if (rank == 0)
            {
                while (idx >= 0 && !(blocks[idx].r > blocks[idx].l))
                {
                    --idx;
                }

                one_block_step_emit_ops(B, blocks[idx].l, blocks[idx].r, u_step_ops, v_step_ops);
                --idx;
            }

            bcast_row_rotations(u_step_ops, 0, MPI_COMM_WORLD);
            bcast_row_rotations(v_step_ops, 0, MPI_COMM_WORLD);

            apply_row_rotations_local(Ut_local, u_step_ops);
            apply_row_rotations_local(Vt_local, v_step_ops);
        }
    }

    // 各进程把自己负责的列块回收给 rank 0
    Matrix Ut_gathered = gather_col_blocks_to_root(Ut_local, m, m, 0, MPI_COMM_WORLD);
    Matrix Vt_gathered = gather_col_blocks_to_root(Vt_local, n, n, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        U = transpose_copy(Ut_gathered);
        V = transpose_copy(Vt_gathered);

        // 迭代结束后统一结构清理与标准化输出。
        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
        {
            B.at(i, i + 1) = 0.0;
        }
        make_nonnegative_and_sort(U, B, V);
    }

    int conv_int = converged ? 1 : 0;
    MPI_Bcast(&conv_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return conv_int != 0;
}
