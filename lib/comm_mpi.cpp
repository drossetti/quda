#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <csignal>
#include <quda_internal.h>
#include <comm_quda.h>

#include "comm_async.h"

#define MPI_CHECK(mpi_call) do {                    \
  int status = mpi_call;                            \
  if (status != MPI_SUCCESS) {                      \
    char err_string[128];                           \
    int err_len;                                    \
    MPI_Error_string(status, err_string, &err_len); \
    err_string[127] = '\0';                         \
    errorQuda("(MPI) %s", err_string);              \
  }                                                 \
} while (0)


typedef enum MsgType {
  MSG_NONE,
  MSG_RECV,
  MSG_SEND,
  MSG_NUM_TYPES
} MsgType_t;


struct MsgHandle_s {
  /**
     The persistant MPI communicator handle that is created with
     MPI_Send_init / MPI_Recv_init.
   */
  MPI_Request request;

  /**
     To create a strided communicator, a MPI_Vector datatype has to be
     created.  This is where it is stored.
   */
  MPI_Datatype datatype;

  /**
     Whether a custom datatype has been created or not.  Used to
     determine whether we need to free the datatype or not.
   */
  bool custom;
#ifdef GPU_ASYNC
  /**
     Async fields
   */
  bool     is_async;
  void    *buffer;
  size_t   nbytes;
  int      rank;
  async_reg_t reg;
  // req is also buffered and tracked down in async
  async_request_t req;
  MsgType_t   type;
#endif
};

static int rank = -1;
static int size = -1;
static int gpuid = -1;
static bool peer2peer_enabled[2][4] = { {false,false,false,false},
                                        {false,false,false,false} };
static bool peer2peer_init = false;

static char partition_string[16] = ",comm=";

#ifdef GPU_ASYNC
static bool async_enabled = false;
#endif

void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  int initialized;
  MPI_CHECK( MPI_Initialized(&initialized) );

  if (!initialized) {
    errorQuda("MPI has not been initialized");
  }

  MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
  MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &size) );

  int grid_size = 1;
  for (int i = 0; i < ndim; i++) {
    grid_size *= dims[i];
  }
  if (grid_size != size) {
    errorQuda("Communication grid size declared via initCommsGridQuda() does not match"
              " total number of MPI ranks (%d != %d)", grid_size, size);
  }

  Topology *topo = comm_create_topology(ndim, dims, rank_from_coords, map_data);
  comm_set_default_topology(topo);

  // determine which GPU this MPI rank will use
  char *hostname = comm_hostname();
  char *hostname_recv_buf = (char *)safe_malloc(128*size);

  MPI_CHECK( MPI_Allgather(hostname, 128, MPI_CHAR, hostname_recv_buf, 128, MPI_CHAR, MPI_COMM_WORLD) );

  gpuid = 0;
  for (int i = 0; i < rank; i++) {
    if (!strncmp(hostname, &hostname_recv_buf[128*i], 128)) {
      gpuid++;
    }
  }

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    errorQuda("No CUDA devices found");
  }
  if (gpuid >= device_count) {
    char *enable_mps_env = getenv("QUDA_ENABLE_MPS");
    if (enable_mps_env && strcmp(enable_mps_env,"1") == 0) {
      gpuid = gpuid%device_count;
      printf("MPS enabled, rank=%d -> gpu=%d\n", comm_rank(), gpuid);
    } else {
      errorQuda("Too few GPUs available on %s", comm_hostname());
    }
  }

  comm_peer2peer_init(hostname_recv_buf);

  host_free(hostname_recv_buf);

  char comm[5];
  comm[0] = (comm_dim_partitioned(0) ? '1' : '0');
  comm[1] = (comm_dim_partitioned(1) ? '1' : '0');
  comm[2] = (comm_dim_partitioned(2) ? '1' : '0');
  comm[3] = (comm_dim_partitioned(3) ? '1' : '0');
  comm[4] = '\0';
  strcat(partition_string, comm);

#ifdef GPU_ASYNC
  {
    int dev_id = 0;
    cudaError_t ret = cudaSuccess;
    ret = cudaGetDevice(&dev_id);
    printf("dev_id=%d ret=%d\n", dev_id, ret);
    if (dev_id >= 0) {
      cudaFree(0);
      ASYNC_CHECK( async_init(MPI_COMM_WORLD) );
    }
  }
#endif
}

void comm_peer2peer_init(const char* hostname_recv_buf)
{
  if (peer2peer_init) return;

  bool disable_peer_to_peer = false;
  char *enable_peer_to_peer_env = getenv("QUDA_ENABLE_P2P");
  if (enable_peer_to_peer_env && strcmp(enable_peer_to_peer_env, "0") == 0) {
    if (getVerbosity() > QUDA_SILENT) printfQuda("Disabling peer-to-peer access\n");
    disable_peer_to_peer = true;
  }

  if (!peer2peer_init && !disable_peer_to_peer) {

    // first check that the local GPU supports UVA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,gpuid);
    if(!prop.unifiedAddressing) return;

    comm_set_neighbor_ranks();

    char *hostname = comm_hostname();

    int *gpuid_recv_buf = (int *)safe_malloc(sizeof(int)*size);

    // There are more efficient ways to do the following,
    // but it doesn't really matter since this function should be
    // called just once.
    MPI_CHECK( MPI_Allgather(&gpuid, 1, MPI_INT, gpuid_recv_buf, 1, MPI_INT, MPI_COMM_WORLD) );

    for(int dir=0; dir<2; ++dir){ // backward/forward directions
      for(int dim=0; dim<4; ++dim){
	int neighbor_rank = comm_neighbor_rank(dir,dim);
	if(neighbor_rank == rank) continue;

	// if the neighbors are on the same
	if (!strncmp(hostname, &hostname_recv_buf[128*neighbor_rank], 128)) {
	  int neighbor_gpuid = gpuid_recv_buf[neighbor_rank];
	  int canAccessPeer[2];
	  cudaDeviceCanAccessPeer(&canAccessPeer[0], gpuid, neighbor_gpuid);
	  cudaDeviceCanAccessPeer(&canAccessPeer[1], neighbor_gpuid, gpuid);

	  if(canAccessPeer[0]*canAccessPeer[1]){
	    peer2peer_enabled[dir][dim] = true;
	    if (getVerbosity() > QUDA_SILENT)
	      printf("Peer-to-peer enabled for rank %d (gpu=%d) with neighbor %d (gpu=%d) dir=%d, dim=%d\n",
		     comm_rank(), gpuid, neighbor_rank, neighbor_gpuid, dir, dim);
	  }
	} // on the same node
      } // different dimensions - x, y, z, t
    } // different directions - forward/backward

    host_free(gpuid_recv_buf);
  }

  peer2peer_init = true;

  // set gdr enablement
  if (comm_gdr_enabled()) {
    printfQuda("Enabling GPU-Direct RDMA access\n");
  } else {
    printfQuda("Disabling GPU-Direct RDMA access\n");
  }

  checkCudaError();
  return;
}


bool comm_peer2peer_enabled(int dir, int dim){
  return peer2peer_enabled[dir][dim];
}


int comm_rank(void)
{
  return rank;
}


int comm_size(void)
{
  return size;
}


int comm_gpuid(void)
{
  return gpuid;
}


static const int max_displacement = 4;

static void check_displacement(const int displacement[], int ndim) {
  for (int i=0; i<ndim; i++) {
    if (abs(displacement[i]) > max_displacement){
      errorQuda("Requested displacement[%d] = %d is greater than maximum allowed", i, displacement[i]);
    }
  }
}

/**
 * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();
  int ndim = comm_ndim(topo);
  check_displacement(displacement, ndim);

  int rank = comm_rank_displaced(topo, displacement);

  int tag = 0;
  for (int i=ndim-1; i>=0; i--) tag = tag * 4 * max_displacement + displacement[i] + max_displacement;
  tag = tag >= 0 ? tag : 2*pow(4*max_displacement,ndim) + tag;

  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  //printf("%s mh=%p async_enabled=%d\n", __func__, mh, (int)async_enabled);
#ifdef GPU_ASYNC
  //if (async_enabled) {
    //mh->is_async = true;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    mh->rank = rank;
    mh->req  = NULL;
    mh->reg  = NULL;
    mh->type = MSG_SEND;
  //} else {
    mh->is_async = false;
    //mh->type = MSG_NONE;
#endif
    MPI_CHECK( MPI_Send_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &(mh->request)) );
    mh->custom = false;
  //}
  return mh;
}


/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();
  int ndim = comm_ndim(topo);
  check_displacement(displacement,ndim);

  int rank = comm_rank_displaced(topo, displacement);

  int tag = 0;
  for (int i=ndim-1; i>=0; i--) tag = tag * 4 * max_displacement - displacement[i] + max_displacement;
  tag = tag >= 0 ? tag : 2*pow(4*max_displacement,ndim) + tag;

  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  //printf("%s mh=%p async_enabled=%d\n", __func__, mh, (int)async_enabled);
#ifdef GPU_ASYNC
  //if (async_enabled) {
    //mh->is_async = true;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    mh->rank = rank;
    mh->req  = NULL;
    mh->reg  = NULL;
    mh->type = MSG_RECV;
  //} else {
    mh->is_async = false;
    //mh->type = MSG_NONE;
#endif
    MPI_CHECK( MPI_Recv_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &(mh->request)) );
    mh->custom = false;
  //}
  return mh;
}


/**
 * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_strided_send_displaced(void *buffer, const int displacement[],
					       size_t blksize, int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();
  int ndim = comm_ndim(topo);
  check_displacement(displacement, ndim);

  int rank = comm_rank_displaced(topo, displacement);

  int tag = 0;
  for (int i=ndim-1; i>=0; i--) tag = tag * 4 * max_displacement + displacement[i] + max_displacement;
  tag = tag >= 0 ? tag : 2*pow(4*max_displacement,ndim) + tag;

  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  // create a new strided MPI type
  MPI_CHECK( MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)) );
  MPI_CHECK( MPI_Type_commit(&(mh->datatype)) );
  mh->custom = true;
#ifdef GPU_ASYNC
  assert (!async_enabled);
  mh->is_async = false;
  mh->req  = NULL;
  mh->reg  = NULL;
  mh->type = MSG_NONE;
#endif
  MPI_CHECK( MPI_Send_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_WORLD, &(mh->request)) );

  return mh;
}


/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[],
						  size_t blksize, int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();
  int ndim = comm_ndim(topo);
  check_displacement(displacement,ndim);

  int rank = comm_rank_displaced(topo, displacement);

  int tag = 0;
  for (int i=ndim-1; i>=0; i--) tag = tag * 4 * max_displacement - displacement[i] + max_displacement;
  tag = tag >= 0 ? tag : 2*pow(4*max_displacement,ndim) + tag;

  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));

  // create a new strided MPI type
  MPI_CHECK( MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)) );
  MPI_CHECK( MPI_Type_commit(&(mh->datatype)) );
  mh->custom = true;
#ifdef GPU_ASYNC
  assert (!async_enabled);
  mh->is_async = false;
  mh->req  = NULL;
  mh->reg  = NULL;
  mh->type = MSG_NONE;
#endif
  MPI_CHECK( MPI_Recv_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_WORLD, &(mh->request)) );

  return mh;
}


void comm_free(MsgHandle *mh)
{
#ifdef GPU_ASYNC
  if (mh->is_async) {
    //TODO make sure we don't leak requests
    //printf("leaking async mh=%p\n", mh);
  }
#endif
  // MPI req is always created, so let's free it
  MPI_CHECK(MPI_Request_free(&(mh->request)));
  if (mh->custom) MPI_CHECK(MPI_Type_free(&(mh->datatype)));

  host_free(mh);
}

#ifdef GPU_ASYNC
int comm_use_prepared()
{
  return async_use_comm_prepared();
}

static int comm_async_start(MsgHandle *mh, CUstream_st *stream)
{
  int ret = 0;
  async_request_t tmp_req;
  switch (mh->type) {
  case MSG_SEND:
    if (stream) {
      ASYNC_CHECK( async_wait_ready_on_stream(mh->rank, stream) );
      ret = async_isend_on_stream(mh->buffer, mh->nbytes, MPI_BYTE, &mh->reg, mh->rank, &mh->req, stream);
    } else {
      ASYNC_CHECK( async_wait_ready(mh->rank) );
      ret = async_isend(mh->buffer, mh->nbytes, MPI_BYTE, &mh->reg, mh->rank, &mh->req);
    }
    break;
  case MSG_RECV:
    ret = async_irecv(mh->buffer, mh->nbytes, MPI_BYTE, &mh->reg, mh->rank, &mh->req);
    if (stream) {
      ret = async_send_ready_on_stream(mh->rank, &tmp_req, stream);
    } else {
      ret = async_send_ready(mh->rank, &tmp_req);
    }
    if (ret) {
        errorQuda("error %d in send_ready\n", ret);
        break;
    }
    break;
  default:
    errorQuda("unsupported MsgHandle type %d\n", mh->type);
    ret = EINVAL;
  }
  return ret;
}

int comm_prepare_start(MsgHandle *mh)
{
  int ret = 0;
  assert(mh->is_async);

  switch (mh->type) {
  case MSG_SEND:
    ASYNC_CHECK( async_prepare_wait_ready(mh->rank) );
    ret = async_prepare_isend(mh->buffer, mh->nbytes, MPI_BYTE, &mh->reg, mh->rank, &mh->req);
    break;
  case MSG_RECV:
    ret = async_irecv(mh->buffer, mh->nbytes, MPI_BYTE, &mh->reg, mh->rank, &mh->req);
    if (ret) {
        errorQuda("error %d in send_ready_on_stream\n", ret);
        break;
    }
    ret = async_prepare_send_ready(mh->rank);
    break;
  default:
    errorQuda("unsupported MsgHandle type %d\n", mh->type);
    ret = EINVAL;
  }
  return ret;
}
#endif

void comm_start(MsgHandle *mh)
{
  //fprintf(stderr, "%s\n", __func__);
  comm_start_on_stream(mh, NULL);
}

void comm_start_on_stream(MsgHandle *mh, CUstream_st *stream)
{
  //fprintf(stderr, "%s mh=%p async_enabled=%d type=%d stream=%p\n", __func__, mh, (int)async_enabled, mh->type, stream);
#ifdef GPU_ASYNC
  if (async_enabled)
    mh->is_async = true;

  assert(mh->type == MSG_SEND || mh->type == MSG_RECV);
  if (mh->is_async) {
    if (comm_use_prepared())
      ASYNC_CHECK( comm_prepare_start(mh) );
    else
      ASYNC_CHECK( comm_async_start(mh, stream) ); 
#else
  if (0) {
#endif
  } else {
    MPI_CHECK( MPI_Start(&(mh->request)) );
  }
  //fprintf(stderr, "%s mh=%p done\n", __func__, mh);
}


void comm_wait(MsgHandle *mh)
{
  //fprintf(stderr, "%s\n", __func__);
  comm_wait_on_stream(mh, NULL);
}


void comm_wait_on_stream(MsgHandle *mh, CUstream_st *stream)
{
  //fprintf(stderr, "%s mh=%p async_enabled=%d type=%d stream=%p\n", __func__, mh, (int)async_enabled, mh->type, stream);
#ifdef GPU_ASYNC
#warning "using GPU_ASYNC"
  if (mh->is_async) {
    assert(mh->type == MSG_SEND || mh->type == MSG_RECV);
    if (comm_use_prepared()) {
      switch(mh->type) {
      case MSG_SEND:
        ASYNC_CHECK( async_prepare_wait_send(&mh->req) );
        break;
      case MSG_RECV:
        ASYNC_CHECK( async_prepare_wait_recv(&mh->req) );
        break;
      default:
        assert(!"invalid type");
        break;
      }
    } else {
      if (stream)
        ASYNC_CHECK( async_wait_all_on_stream(1, &mh->req, stream) );
      else
        ASYNC_CHECK( async_wait_all(1, &mh->req) );
    }
#else
  if (0) {
#endif
  } else {
    // ignore stream
    MPI_CHECK( MPI_Wait(&(mh->request), MPI_STATUS_IGNORE) );
  }
  //fprintf(stderr, "%s mh=%p done\n", __func__, mh);
}


int comm_query(MsgHandle *mh)
{
  int query;
#ifdef GPU_ASYNC
  if (mh->is_async) {
    // TODO implement completion query
    // query == 1 -> comm is completed
    query = 0;
#else
  if (0) {
#endif
  } else {
    MPI_CHECK( MPI_Test(&(mh->request), &query, MPI_STATUS_IGNORE) );
  }
  return query;
}


void comm_allreduce(double* data)
{
  double recvbuf;
  MPI_CHECK( MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
  *data = recvbuf;
}


void comm_allreduce_max(double* data)
{
  double recvbuf;
  MPI_CHECK( MPI_Allreduce(data, &recvbuf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) );
  *data = recvbuf;
}

void comm_allreduce_array(double* data, size_t size)
{
  double *recvbuf = new double[size];
  MPI_CHECK( MPI_Allreduce(data, recvbuf, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) );
  memcpy(data, recvbuf, size*sizeof(double));
  delete []recvbuf;
}


void comm_allreduce_int(int* data)
{
  int recvbuf;
  MPI_CHECK( MPI_Allreduce(data, &recvbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) );
  *data = recvbuf;
}


/**  broadcast from rank 0 */
void comm_broadcast(void *data, size_t nbytes)
{
  MPI_CHECK( MPI_Bcast(data, (int)nbytes, MPI_BYTE, 0, MPI_COMM_WORLD) );
}


void comm_barrier(void)
{
  MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );
}


void comm_abort(int status)
{
#ifdef HOST_DEBUG
  raise(SIGINT);
#endif
  MPI_Abort(MPI_COMM_WORLD, status) ;
}

const char* comm_dim_partitioned_string() {
  return partition_string;
}


#ifdef GPU_ASYNC


void comm_enable_async(int _async_enabled)
{
  async_enabled = !!_async_enabled && async_use_comm_async_stream();
  //printf("async is %s\n", async_enabled ? "enabled" : "disabled");
}


int comm_use_async()
{
  return async_use_comm_async_stream();
}


int comm_flush()
{
  int ret = 0;
  //printfQuda("calling async flush\n");
  int retcode = async_flush();
  if (retcode < 0) {
    errorQuda("error %d in async_flush()\n", retcode);
    ret = retcode;
  }
  // TODO: propagate error from flush
  return ret;
}


int comm_progress()
{
  int ret = 0;
  //printfQuda("calling async progress\n");
  int retcode = async_progress();
  if (retcode < 0) {
    errorQuda("error %d in async_progress()\n", retcode);
    ret = retcode;
  }
  return ret;
}


int comm_prepare_wait(MsgHandle *mh)
{
  int ret = 0;
  assert(comm_use_prepared());
  assert(mh->is_async);
  switch(mh->type) {
  case MSG_SEND:
    ret = async_prepare_wait_send(&mh->req);
    break;
  case MSG_RECV:
    ret = async_prepare_wait_recv(&mh->req);
    break;
  default:
    errorQuda("unsupported MsgHandle type %d\n", mh->type);
    ret = EINVAL;
  }
  return ret;
}

int comm_prepare_wait_value32(unsigned int *pw, unsigned int value, int flags)
{
  return async_prepare_wait_value32(pw, value, flags);
}

int comm_prepare_write_value32(unsigned int *pw, unsigned int value)
{
  return async_prepare_write_value32(pw, value);
}

int comm_flush_prepared(CUstream_st *stream)
{
  int ret = 0;
  //fprintf(stderr, "here\n");
  if (comm_use_prepared()) {
    //fprintf(stderr, "there\n");
    assert(stream);
    ret = async_submit_prepared(stream);
  }
  return ret;
}


#endif // GPU_ASYNC
