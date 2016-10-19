#pragma once

#define __ASYNC_CHECK(stmt, cond_str)                    \
    do {                                \
            int result = (stmt);                                        \
            if (result) {                                               \
                fprintf(stderr, "%s failed at %s:%d error=%d\n",        \
                        cond_str, __FILE__, __LINE__, result);          \
                exit(EXIT_FAILURE);                                     \
            }                                                           \
        } while (0)

#define ASYNC_CHECK(stmt) __ASYNC_CHECK(stmt, #stmt)

#ifdef __cplusplus
extern "C" {
#endif

    int async_use_comm();
    int async_use_gdrdma();
    int async_use_async();
    int async_use_gpu_comm();

    typedef struct async_request  *async_request_t;
    typedef struct async_reg      *async_reg_t;
    typedef struct CUstream_st   *async_stream_t;
    int async_init(MPI_Comm comm);
    int async_send_ready_on_stream(int rank, async_request_t *creq, async_stream_t stream);
    int async_send_ready(int rank, async_request_t *creq);
    int async_wait_ready_on_stream(int rank, async_stream_t stream);
    int async_wait_ready(int rank);
    int async_test_ready(int rank, int *p_rdy);
    int async_irecv(void *recv_buf, size_t size, MPI_Datatype type, async_reg_t *reg, int src_rank, 
                   async_request_t *req);
    int async_isend_on_stream(void *send_buf, size_t size, MPI_Datatype type, async_reg_t *reg,
                             int dest_rank, async_request_t *req, async_stream_t stream);
    int async_isend(void *send_buf, size_t size, MPI_Datatype type, async_reg_t *reg,
                   int dest_rank, async_request_t *req);
    int async_wait_all(int count, async_request_t *creqs);
    int async_wait_all_on_stream(int count, async_request_t *creqs, async_stream_t stream);
    int async_wait(async_request_t *creq);
    int async_flush();
    int async_progress();

    int async_prepare_wait_ready(int rank);
    int async_prepare_isend(void *send_buf, size_t size, MPI_Datatype type, async_reg_t *creg,
                           int dest_rank, async_request_t *creq);
    int async_prepare_wait_all(int count, async_request_t *creqs);
    int async_register(void *buf, size_t size, async_reg_t *creg);
    
#ifdef __cplusplus
}
#endif
