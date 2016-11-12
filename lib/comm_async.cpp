#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "comm_async.h"
#include "mp.h"

// debug stuff

int dbg_enabled()
{
    static int dbg_is_enabled = -1;
    if (-1 == dbg_is_enabled) {        
        const char *env = getenv("QUDA_ASYNC_ENABLE_DEBUG");
        if (env) {
            int en = atoi(env);
            dbg_is_enabled = !!en;
            printf("QUDA_ASYNC_ENABLE_DEBUG=%s\n", env);
        } else
            dbg_is_enabled = 0;
    }
    return dbg_is_enabled;
}

#define DBG(FMT, ARGS...)                                               \
    do {                                                                \
        if (dbg_enabled()) {                                            \
            fprintf(stderr, "[%d] [%d] ASYNC %s(): " FMT,               \
                    getpid(), async_rank, __FUNCTION__ , ## ARGS);      \
            fflush(stderr);                                             \
        }                                                               \
    } while(0)

#define MP_CHECK(stmt)                                  \
do {                                                    \
    int result = (stmt);                                \
    if (0 != result) {                                  \
        fprintf(stderr, "[%s:%d] mp call failed \n",    \
         __FILE__, __LINE__);                           \
        exit(-1);                                       \
    }                                                   \
    assert(0 == result);                                \
} while (0)


#define async_err(FMT, ARGS...)  do { fprintf(stderr, "ERR [%d] %s() " FMT, async_rank, __FUNCTION__ , ## ARGS); fflush(stderr); } while(0)


#ifndef MAX
#define MAX(A,B) ((A)>(B)?(A):(B))
#endif

#define MAX_RANKS 128

static int         async_initialized = 0;
static int         rank_to_peer[MAX_RANKS] = {0,};
static int         peers[MAX_RANKS] = {0,};
static int         n_peers = -1;
static const int   bad_peer = -1;
static int         async_size;
static int         async_rank;

// tables are indexed by rank, not peer
static uint32_t   *ready_table;
static mp_reg_t    ready_table_reg;
static mp_window_t ready_table_win;

static uint32_t   *ready_values;
static uint32_t   *remote_ready_values;
static mp_reg_t    remote_ready_values_reg;

#define MAX_REQS 16384
static mp_request_t reqs[MAX_REQS];
static int n_reqs = 0;

#define PAGE_BITS 12
#define PAGE_SIZE (1ULL<<PAGE_BITS)
#define PAGE_OFF  (PAGE_SIZE-1)
#define PAGE_MASK (~(PAGE_OFF))

#define mb()    __asm__ volatile("mfence":::"memory")
#define rmb()   __asm__ volatile("lfence":::"memory")
#define wmb()   __asm__ volatile("sfence" ::: "memory")
#define iomb() mb()

#define ACCESS_ONCE(V)                          \
    (*(volatile __typeof__ (V) *)&(V))

static inline void arch_pause(void)
{
        __asm__ volatile("pause\n": : :"memory");
}


static inline void arch_cpu_relax(void)
{
        //rmb();
        //arch_rep_nop();
        arch_pause();
        //arch_pause(); arch_pause();
        //BUG: poll_lat hangs after a few iterations
        //arch_wait();
}

int async_use_comm_mp()
{
    static int use_comm = -1;
    if (-1 == use_comm) {
        const char *env = getenv("QUDA_USE_COMM_MP");
        if (env) {
            use_comm = !!atoi(env);
            printf("WARNING: %s libMP communications\n", (use_comm)?"enabling":"disabling");
        } else
            use_comm = 0; // default
    }
    return use_comm;
}

int async_use_comm_rdma()
{
    static int use_gdrdma = -1;
    if (-1 == use_gdrdma) {
        const char *env = getenv("QUDA_USE_COMM_RDMA");
        if (env) {
            use_gdrdma = !!atoi(env);
            printf("WARNING: %s RDMA communications\n", (use_gdrdma)?"enabling":"disabling");
        } else
            use_gdrdma = 0; // default
    }
    return use_gdrdma;
}

int async_use_comm_async_kernel()
{
    static int use_gpu_comm = -1;
    if (-1 == use_gpu_comm) {
        const char *env = getenv("QUDA_USE_COMM_ASYNC_KERNEL");
        if (env) {
            use_gpu_comm = !!atoi(env);
            printf("WARNING: %s GPU-initiated kernel-synchronous communications\n", (use_gpu_comm)?"enabling":"disabling");
        } else
            use_gpu_comm = 0; // default
    }
    return use_gpu_comm;
}

int async_use_comm_async_stream()
{
    static int use_async = -1;
    if (-1 == use_async) {
        const char *env = getenv("QUDA_USE_COMM_ASYNC_STREAM");
        if (env) {
            use_async = !!atoi(env);
            printf("WARNING: %s GPUDirect Async Stream-synchronous communications\n", (use_async)?"enabling":"disabling");
        } else
            use_async = 0; // default
    }
    return use_async;
}

int async_use_comm_prepared()
{
    static int use_prepared = -1;
    if (-1 == use_prepared) {
        const char *env = getenv("QUDA_USE_COMM_ASYNC_PREPARED");
        if (env) {
            use_prepared = !!atoi(env);
            printf("WARNING: %s GPUDirect Async prepared Stream-synchronous communications\n", (use_prepared)?"enabling":"disabling");
        } else
            use_prepared = 0; // default
    }
    return use_prepared;
}

int async_init(MPI_Comm comm)
{
    int i, j;

    if (async_initialized) {
        async_err("async_init called twice\n");
        return 1;
    }

    MPI_Comm_size(comm, &async_size);
    MPI_Comm_rank(comm, &async_rank);

    assert(async_size < MAX_RANKS);

    // init peers    
    for (i=0, j=0; i<async_size; ++i) {
        if (i!=async_rank) {
            peers[j] = i;
            rank_to_peer[i] = j;
            DBG("peers[%d]=rank %d\n", j, i);
            ++j;
        } else {
            // self reference is forbidden
            rank_to_peer[i] = bad_peer;
        }
    }
    n_peers = j;
    assert(async_size-1 == n_peers);
    DBG("n_peers=%d\n", n_peers);

    int mp_ver = 0;
    MP_CHECK(mp_query_param(MP_PARAM_VERSION, &mp_ver));
    if (!MP_API_VERSION_COMPATIBLE(mp_ver)) {
        async_err("incompatible libmp version\n");
        return 1;
    }

    MP_CHECK(mp_init(comm, peers, n_peers
#if MP_API_MAJOR_VERSION > 1
                     , MP_INIT_DEFAULT
#endif
                     ));

    // init ready stuff
    size_t table_size = MAX(sizeof(*ready_table) * async_size, PAGE_SIZE);
    ready_table = (uint32_t *)memalign(PAGE_SIZE, table_size);
    assert(ready_table);

    ready_values = (uint32_t *)malloc(table_size);
    assert(ready_values);

    remote_ready_values = (uint32_t *)memalign(PAGE_SIZE, table_size);
    assert(remote_ready_values);
    
    for (i=0; i<async_size; ++i) {
        ready_table[i] = 0;  // remotely written table
        ready_values[i] = 1; // locally expected value
        remote_ready_values[i] = 1; // value to be sent remotely
    }
    iomb();

    DBG("registering ready_table size=%zd\n", table_size);
    MP_CHECK(mp_register(ready_table, table_size, &ready_table_reg));
    DBG("creating ready_table window\n");
    MP_CHECK(mp_window_create(ready_table, table_size, &ready_table_win));
    DBG("registering remote_ready_table\n");
    MP_CHECK(mp_register(remote_ready_values, table_size, &remote_ready_values_reg));

    async_initialized = 1;

    return 0;
}

static size_t async_size_of_mpi_type(MPI_Datatype mpi_type)
{
    size_t ret = 0;

    if (mpi_type == MPI_BYTE)
        ret = sizeof(char);
    else if (mpi_type == MPI_DOUBLE)
        ret = sizeof(double);
    else {
        async_err("invalid type\n");
        exit(1);
    }
    return ret;
}

static int async_mpi_rank_to_peer(int rank)
{
    assert(async_initialized);
    assert(rank < async_size);
    assert(rank >= 0);
    int peer = rank;
    //int peer = rank_to_peer[rank];
    //assert(peer != bad_peer);
    //assert(peer < n_peers);
    return peer;
}

static void atomic_inc(uint32_t *ptr)
{
    __sync_fetch_and_add(ptr, 1);
    //++ACCESS_ONCE(*ptr);
    iomb();
}

static void async_track_request(mp_request_t *req)
{
    assert(n_reqs < MAX_REQS);    
    reqs[n_reqs++] = *req;
    DBG("n_reqs=%d\n", n_reqs);
}

int async_send_ready_on_stream(int rank, async_request_t *creq, cudaStream_t stream)
{
    assert(async_initialized);
    assert(rank < async_size);
    int ret = 0;
    int peer = async_mpi_rank_to_peer(rank);
    mp_request_t *req = (mp_request_t*)creq;
    assert(req);
    int remote_offset = /*self rank*/async_rank * sizeof(uint32_t);
    DBG("dest_rank=%d payload=%x offset=%d\n", rank, remote_ready_values[rank], remote_offset);
    MP_CHECK(mp_iput_on_stream(&remote_ready_values[rank], sizeof(uint32_t), &remote_ready_values_reg, 
                               peer, remote_offset, &ready_table_win, req, MP_PUT_INLINE, stream));
    //MP_CHECK(mp_wait(req));
    async_track_request(req);
    atomic_inc(&remote_ready_values[rank]);
    return ret;
}

int async_send_ready(int rank, async_request_t *creq)
{
    assert(async_initialized);
    assert(rank < async_size);
    int ret = 0;
    int peer = async_mpi_rank_to_peer(rank);
    mp_request_t *req = (mp_request_t*)creq;
    assert(req);
    int remote_offset = /*my rank*/async_rank * sizeof(uint32_t);
    DBG("dest_rank=%d payload=%x offset=%d\n", rank, remote_ready_values[rank], remote_offset);
    MP_CHECK(mp_iput(&remote_ready_values[rank], sizeof(uint32_t), &remote_ready_values_reg, 
                     peer, remote_offset, &ready_table_win, req, MP_PUT_INLINE));
    //MP_CHECK(mp_wait(req));
    async_track_request(req);
    atomic_inc(&remote_ready_values[rank]);
    return ret;
}

int async_wait_ready_on_stream(int rank, cudaStream_t stream)
{
    assert(async_initialized);
    assert(rank < async_size);
    int ret = 0;
    int peer = async_mpi_rank_to_peer(rank);
    DBG("rank=%d payload=%x\n", rank, ready_values[rank]); 
    MP_CHECK(mp_wait_dword_geq_on_stream(&ready_table[rank], ready_values[rank], stream));
    ready_values[rank]++;
    return ret;
}

int async_wait_ready(int rank)
{
    assert(async_initialized);
    assert(rank < async_size);
    int ret = 0;
    int peer = async_mpi_rank_to_peer(rank);
    int cnt = 0;
    DBG("rank=%d payload=%x\n", rank, ready_values[rank]);
    while (ACCESS_ONCE(ready_table[rank]) < ready_values[rank]) {
        rmb();
        arch_cpu_relax();
        ++cnt;
        if (cnt > 10000) {
            async_progress();
            cnt = 0;
        }
    }
    ready_values[rank]++;
    return ret;
}

int async_test_ready(int rank, int *p_rdy)
{
    assert(async_initialized);
    assert(rank < async_size);
    int ret = 0;
    int peer = async_mpi_rank_to_peer(rank);
    static int cnt = 0;
    DBG("rank=%d payload=%x\n", rank, ready_values[rank]);
    do {
        rmb();
        *p_rdy = !(ACCESS_ONCE(ready_table[rank]) < ready_values[rank]);
        if (*p_rdy) {
            ++ready_values[rank];
            break;
        }
        ++cnt;
        if (cnt > 10000) {
            arch_cpu_relax();
            cnt = 0;
        }
    } while(0);
    return ret;
}

int async_wait_all_on_stream(int count, async_request_t *creqs, cudaStream_t stream)
{
    int ret = 0;
    DBG("count=%d\n", count);
    assert(async_initialized);
    mp_request_t *reqs = (mp_request_t*)creqs;
    if (1 == count) {
        assert(*reqs);
        MP_CHECK(mp_wait_on_stream(reqs, stream));
    } else {
        MP_CHECK(mp_wait_all_on_stream(count, reqs, stream));
    }
    memset(creqs, 0, sizeof(async_request_t)*count);
    return ret;
}

int async_wait_all(int count, async_request_t *creqs)
{
    int ret = 0;
    DBG("count=%d\n", count);
    assert(async_initialized);
    mp_request_t *reqs = (mp_request_t*)creqs;
    MP_CHECK(mp_wait_all(count, reqs));
    memset(creqs, 0, sizeof(async_request_t)*count);
    return ret;
}

int async_wait(async_request_t *creq)
{
    int ret = 0;
    assert(async_initialized);
    mp_request_t *req = (mp_request_t*)creq;
    MP_CHECK(mp_wait(req));
    memset(creq, 0, sizeof(async_request_t));
    return ret;
}

// tags are not supported!!!
int async_irecv(void *recv_buf, size_t size, MPI_Datatype type, async_reg_t *creg,
               int src_rank, async_request_t *creq)
{
    assert(async_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*async_size_of_mpi_type(type);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);
    mp_request_t *req = (mp_request_t*)creq;
    assert(req);
    int peer = async_mpi_rank_to_peer(src_rank);

    DBG("src_rank=%d peer=%d nbytes=%zd buf=%p *reg=%p\n", src_rank, peer, nbytes, recv_buf, *reg);

    if (!size) {
        ret = -EINVAL;
        async_err("SIZE==0\n");
        goto out;
    }

    if (!*reg) {
        DBG("registering buffer %p\n", recv_buf);
        MP_CHECK(mp_register(recv_buf, nbytes, reg));
    }

    retcode = mp_irecv(recv_buf,
                       nbytes,
                       peer,
                       reg,
                       req);
    if (retcode) {
        async_err("error in mp_irecv ret=%d\n", retcode);
        ret = -1;
        goto out;
    }
    async_track_request(req);
out:
    return ret;
}

int async_isend_on_stream(void *send_buf, size_t size, MPI_Datatype type, async_reg_t *creg,
                         int dest_rank, async_request_t *creq, cudaStream_t stream)
{
    assert(async_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*async_size_of_mpi_type(type);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);
    mp_request_t *req = (mp_request_t*)creq;
    int peer = async_mpi_rank_to_peer(dest_rank);

    DBG("dest_rank=%d peer=%d nbytes=%zd\n", dest_rank, peer, nbytes);

    if (!size) {
        ret = -EINVAL;
        async_err("SIZE==0\n");
        goto out;
    }

    if (!*reg) {
        DBG("registering buffer %p\n", send_buf);
        MP_CHECK(mp_register(send_buf, nbytes, reg));
    }
    retcode = mp_isend_on_stream(send_buf,
                                 nbytes,
                                 peer,
                                 reg,
                                 req,
                                 stream);
    if (retcode) {
        async_err("error in mp_isend_on_stream ret=%d\n", retcode);
        ret = -1;
        goto out;
    }
    async_track_request(req);
out:
    return ret;
}

int async_isend(void *send_buf, size_t size, MPI_Datatype type, async_reg_t *creg,
               int dest_rank, async_request_t *creq)
{
    assert(async_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*async_size_of_mpi_type(type);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);
    mp_request_t *req = (mp_request_t*)creq;
    int peer = async_mpi_rank_to_peer(dest_rank);

    DBG("dest_rank=%d peer=%d nbytes=%zd\n", dest_rank, peer, nbytes);

    if (!size) {
        ret = -EINVAL;
        async_err("SIZE==0\n");
        goto out;
    }

    if (!*reg) {
        DBG("registering buffer %p\n", send_buf);
        MP_CHECK(mp_register(send_buf, nbytes, reg));
    }

    retcode = mp_isend(send_buf,
                       nbytes,
                       peer,
                       reg,
                       req);
    if (retcode) {
        async_err("error in mp_isend ret=%d\n", retcode);
        ret = -1;
        goto out;
    }
    async_track_request(req);
out:
    return ret;
}

int async_register(void *buf, size_t size, async_reg_t *creg)
{
    assert(async_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*async_size_of_mpi_type(MPI_DOUBLE);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);

    if (!size) {
        ret = -EINVAL;
        async_err("SIZE==0\n");
        goto out;
    }

    if (!*reg) {
        DBG("registering buffer %p\n", buf);
        MP_CHECK(mp_register(buf, nbytes, reg));
    }

out:
    return ret;
}


int async_flush()
{
    int ret = 0;
    DBG("n_reqs=%d\n", n_reqs);
    assert(n_reqs < MAX_REQS);
#if 0
    do {
        rmb();
        uint32_t w0 = ACCESS_ONCE(ready_table[0]);
        uint32_t w1 = ACCESS_ONCE(ready_table[1]);
        DBG("ready_table: %08x %08x\n", w0, w1);
        ret = mp_progress_all(n_reqs, reqs);
        arch_cpu_relax();
        cudaStreamQuery(NULL);
    } while(ret < n_reqs);
#endif
    ret = mp_wait_all(n_reqs, reqs);
    if (ret) {
        async_err("got error in mp_wait_all ret=%d\n", ret);
        exit(EXIT_FAILURE);
    }
    n_reqs = 0;
    return ret;
}

int async_progress()
{
    DBG("n_reqs=%d\n", n_reqs);
    assert(n_reqs < MAX_REQS);
    int ret = mp_progress_all(n_reqs, reqs);
    if (ret < 0) {
        async_err("ret=%d\n", ret);
    }
    return ret;
}

static struct desc_queue {
    mp_desc_queue_t mdq;
    desc_queue() {
        MP_CHECK(mp_desc_queue_alloc(&mdq));
    }
    ~desc_queue() {
        MP_CHECK(mp_desc_queue_free(&mdq));
    }
    mp_desc_queue_t *operator&() {
        if (!mdq)
        return &mdq;
    }
} dq;


int async_prepare_isend(void *send_buf, size_t size, MPI_Datatype type, async_reg_t *creg, int dest_rank, async_request_t *creq)
{
    assert(async_initialized);
    int ret = 0;
    int retcode;
    size_t nbytes = size*async_size_of_mpi_type(type);
    mp_reg_t *reg = (mp_reg_t*)creg;
    assert(reg);
    mp_request_t *req = (mp_request_t*)creq;
    assert(req);
    int peer = async_mpi_rank_to_peer(dest_rank);
    DBG("dest_rank=%d peer=%d nbytes=%zd\n", dest_rank, peer, nbytes);
    if (!size) {
        ret = -EINVAL;
        async_err("SIZE==0\n");
        goto out;
    }
    if (!*reg) {
        DBG("registering buffer %p\n", send_buf);
        MP_CHECK(mp_register(send_buf, nbytes, reg));
    }
    MP_CHECK(mp_send_prepare(send_buf, nbytes, peer, reg, req));
    MP_CHECK(mp_desc_queue_add_send(&dq, req));
    async_track_request(req);
out:
    return ret;
}

int async_prepare_wait_ready(int rank)
{
    assert(async_initialized);
    assert(rank < async_size);
    int ret = 0;
    int peer = async_mpi_rank_to_peer(rank);
    DBG("rank=%d\n", rank);
    MP_CHECK(mp_desc_queue_add_wait_value32(&dq, &ready_table[rank], ready_values[rank], MP_WAIT_GEQ));
    //async_track_request(req);
    ready_values[rank]++;
    return ret;
}

int async_prepare_send_ready(int rank, async_request_t *creq)
{
    assert(async_initialized);
    assert(rank < async_size);
    int ret = 0;
    int peer = async_mpi_rank_to_peer(rank);
    mp_request_t *req = (mp_request_t *)creq;
    int remote_offset = /*self rank*/async_rank * sizeof(uint32_t);
    DBG("dest_rank=%d payload=%x offset=%d\n", rank, remote_ready_values[rank], remote_offset);
    MP_CHECK(mp_put_prepare(&remote_ready_values[rank], sizeof(uint32_t), &remote_ready_values_reg, 
                            peer, remote_offset, &ready_table_win, req, MP_PUT_INLINE));
    MP_CHECK(mp_desc_queue_add_send(&dq, req));
    async_track_request(req);
    atomic_inc(&remote_ready_values[rank]);
    return ret;
}

int async_prepare_wait_send(async_request_t *creq)
{
    assert(async_initialized);
    int ret = 0;
    mp_request_t *req = (mp_request_t *)creq;
    MP_CHECK(mp_desc_queue_add_wait_send(&dq, req));
    //async_track_desc(desc);
    return ret;
}

int async_prepare_wait_recv(async_request_t *creq)
{
    assert(async_initialized);
    int ret = 0;
    mp_request_t *req = (mp_request_t *)creq;
    MP_CHECK(mp_desc_queue_add_wait_recv(&dq, req));
    //async_track_desc(desc);
    return ret;
}

int async_prepare_wait_value32(uint32_t *pw, uint32_t value, int flags)
{
    assert(async_initialized);
    int ret = 0;
    MP_CHECK(mp_desc_queue_add_wait_value32(&dq, pw, value, flags));
    return ret;
}

int async_prepare_write_value32(uint32_t *pw, uint32_t value)
{
    assert(async_initialized);
    int ret = 0;
    MP_CHECK(mp_desc_queue_add_write_value32(&dq, pw, value));
    return ret;
}

int async_submit_prepared(async_stream_t stream)
{
    //assert(n_descs < MAX_DESCS);
    //memset(preqs, 0, sizeof(descs[0])*n_descs);
    //n_preqs = 0;

    // flush and invalidate desc queue
    MP_CHECK(mp_desc_queue_post_on_stream(stream, &dq, 0));
}
