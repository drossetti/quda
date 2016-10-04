#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <quda_internal.h>
#include <comm_quda.h>

#include <mp.h>

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

#define DBG(FMT, ...)																										\
	do {																																	\
		if (true || getVerbosity() == QUDA_DEBUG_VERBOSE)										\
			fprintf(stderr, "[%d] %s:%d " FMT, rank, __FUNCTION__, __LINE__, __VA_ARGS__); \
	} while(0)


struct MsgHandle_s {
  MPI_Request request;
  MPI_Datatype datatype;

  int rank;
  int peer;
  enum { TYPE_RECV, TYPE_SEND, TYPE_STRIDED_RECV, TYPE_STRIDED_SEND } type;
  bool gdsync;
  void *buffer;
  size_t nbytes;
  // for TYPE_STRIDED_*
  size_t blksize;
  int nblocks;
  size_t stride;
  mp_request_t req;
    mp_reg_t mem_reg;
};

static int rank = -1;
static int size = -1;
static int gpuid = -1;

static int peer_count = 0;
static int peers[QUDA_MAX_DIM];
static int cart_coords[QUDA_MAX_DIM];
static int cart_rank;

static bool gdsync_enabled = false;

void comm_enable_gdsync(bool enabled)
{
	gdsync_enabled = enabled;
}

bool comm_gdsync_enabled()
{
	return gdsync_enabled;
}


void comm_init(int ndim, const int *dims, QudaCommsMap rank_from_coords, void *map_data)
{
  int initialized;
  MPI_CHECK( MPI_Initialized(&initialized) );

  warningQuda("in gdsync comm_init");


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
  host_free(hostname_recv_buf);

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    errorQuda("No CUDA devices found");
  }
  if (gpuid >= device_count) {
    errorQuda("Too few GPUs available on %s", hostname);
  }

  /*setup IB communication infrastructure*/

  int npdim = 0;
  int period[QUDA_MAX_DIM];
  int pdims[QUDA_MAX_DIM];
  for (int i = 0, pi = 0; i < ndim; i++) {
    bool is_partitioned = (dims[i] > 1);
    // let's neglect unpartitioned dims for now
		DBG("is_partitioned=%d dims[%d]=%d\n", is_partitioned, i, dims[i]);
    if (is_partitioned) {
      // add 2 peers for every partitioned dimension, even when dims[i] == 2
      peer_count += 2;
      pdims[pi] = dims[i];
      period[pi] = 1;
			DBG("dim:%d pdim=%d pdims[]=%d period=%d\n", i, pi, pdims[i], period[pi]);
      pi += 1;
      npdim += 1;
    }
  }

  int reorder = 0;
  MPI_Comm cartcomm;
  MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, npdim, pdims, period, reorder, &cartcomm));
  MPI_CHECK(MPI_Comm_rank(cartcomm, &cart_rank));
  
  for (int i = 0; i < npdim; i++) {
    int prev, next;
    MPI_CHECK(MPI_Cart_shift(cartcomm, i,  1,  &prev, &next ));
    assert(prev != MPI_UNDEFINED);
    assert(next != MPI_UNDEFINED);
    peers[2*i+0] = prev;
    peers[2*i+1] = next;
		DBG("pdim:%d prev=%d next=%d\n", i, prev, next);
  }

  MPI_Cart_coords(cartcomm, cart_rank, npdim, cart_coords);

  MPI_Barrier(MPI_COMM_WORLD);
	DBG("peer_count=%d\n", peer_count);
  int ret = SUCCESS;
  ret = mp_init (cartcomm, peers, peer_count); 
  if (ret != SUCCESS) {
    fprintf(stderr, "mp_init returned error \n");
    exit(-1);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Comm_free (&cartcomm);
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

static int comm_peer_from_displacement(const int displacement[])
{
  int peer = -1;
  for (int i = 0; i < QUDA_MAX_DIM; ++i)
	  if (displacement[i]) {
		  peer = 2*i + displacement[i]<0?0:1;
      break;
    }
  assert(peer >= 0);
  return peer;
}

//#define DBG() do { fprintf(stderr,"%d: %s buffer=%p\n", rank, __FUNCTION__, buffer); } while(0)
#define BUFDBG() DBG("buffer=%p\n", buffer)

/**
 * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_send_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  int tag = comm_rank();
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
  if (comm_gdsync_enabled()) {
		BUFDBG();
    mh->gdsync = true;
    mh->type = MsgHandle_s::TYPE_SEND;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    mh->rank = rank;
    mh->peer = comm_peer_from_displacement(displacement);
    mp_register(buffer, nbytes, &mh->mem_reg);
    assert(mh->mem_reg);
  } else {
    mh->gdsync = false;
    MPI_CHECK( MPI_Send_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &(mh->request)) );
  }
  return mh;
}


/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_receive_displaced(void *buffer, const int displacement[], size_t nbytes)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  int tag = rank;
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
  if (comm_gdsync_enabled()) {
		BUFDBG();
    mh->gdsync = true;
    mh->type = MsgHandle_s::TYPE_RECV;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    mh->rank = rank;
    mh->peer = comm_peer_from_displacement(displacement);
    mp_register(buffer, nbytes, &mh->mem_reg);
    assert(mh->mem_reg);
  } else {
    mh->gdsync = false;
    MPI_CHECK( MPI_Recv_init(buffer, nbytes, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &(mh->request)) );
  }
  return mh;
}


/**
 * Declare a message handle for sending to a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_strided_send_displaced(void *buffer, const int displacement[],
					       size_t blksize, int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  int tag = comm_rank();
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
  if (comm_gdsync_enabled()) {
		BUFDBG();
    mh->gdsync = true;
    mh->type = MsgHandle_s::TYPE_STRIDED_SEND;
    mh->buffer = buffer;
    mh->nbytes = 0;
    mh->blksize = blksize;
    mh->nblocks = nblocks;
    mh->stride = stride;
    mh->rank = rank;
    mh->peer = comm_peer_from_displacement(displacement);
    assert(blksize <= stride);
    mp_register(buffer, stride*nblocks, &mh->mem_reg);
    assert(mh->mem_reg);
  } else {
    mh->gdsync = false;
    // create a new strided MPI type
    MPI_CHECK( MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)) );
    MPI_CHECK( MPI_Type_commit(&(mh->datatype)) );
    MPI_CHECK( MPI_Send_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_WORLD, &(mh->request)) );
  }

  return mh;
}


/**
 * Declare a message handle for receiving from a node displaced in (x,y,z,t) according to "displacement"
 */
MsgHandle *comm_declare_strided_receive_displaced(void *buffer, const int displacement[],
						  size_t blksize, int nblocks, size_t stride)
{
  Topology *topo = comm_default_topology();

  int rank = comm_rank_displaced(topo, displacement);
  int tag = rank;
  MsgHandle *mh = (MsgHandle *)safe_malloc(sizeof(MsgHandle));
  if (comm_gdsync_enabled()) {
		BUFDBG();
    mh->gdsync = true;
    mh->type = MsgHandle_s::TYPE_STRIDED_RECV;
    mh->buffer = buffer;
    mh->nbytes = 0;
    mh->blksize = blksize;
    mh->nblocks = nblocks;
    mh->stride = stride;
    mh->rank = rank;
    mh->peer = comm_peer_from_displacement(displacement);
    assert(blksize <= stride);
    mp_register(buffer, stride*nblocks, &mh->mem_reg);
    assert(mh->mem_reg);
  } else {
    // create a new strided MPI type
    MPI_CHECK( MPI_Type_vector(nblocks, blksize, stride, MPI_BYTE, &(mh->datatype)) );
    MPI_CHECK( MPI_Type_commit(&(mh->datatype)) );    
    MPI_CHECK( MPI_Recv_init(buffer, 1, mh->datatype, rank, tag, MPI_COMM_WORLD, &(mh->request)) );
  }
  return mh;
}


void comm_free(MsgHandle *mh)
{
  if (mh->gdsync) {
		mp_deregister(&mh->mem_reg);
	}
	memset(mh, 0, sizeof(mh));
  host_free(mh);
}


void comm_gdsync_start(MsgHandle *mh, cudaStream_t stream)
{
	int ret = 0;

  assert(mh->gdsync);
	
	if (getVerbosity() == QUDA_DEBUG_VERBOSE) printfQuda("[%d] here %s:%d\n", comm_rank(), __FUNCTION__, __LINE__);

  switch(mh->type) {
  case MsgHandle_s::TYPE_SEND:
    ret = mp_isend_on_stream(mh->buffer, mh->nbytes, mh->peer, &mh->mem_reg, &mh->req, stream);
    if (ret)
			errorQuda("error in isend_on_stream %s\n", __FUNCTION__);
    break;
  case MsgHandle_s::TYPE_RECV:
    assert(!stream);
    ret = mp_irecv(mh->buffer, mh->nbytes, mh->peer, &mh->mem_reg, &mh->req);
    if (ret)
			errorQuda("error in irecv %s\n", __FUNCTION__);
    break;
  case MsgHandle_s::TYPE_STRIDED_SEND:
    {
      struct iovec v[mh->nblocks];
      for (int i=0; i<mh->nblocks; ++i) {
        v[i].iov_base = (char*)mh->buffer + i*mh->stride;
        v[i].iov_len = mh->blksize;
      }
      assert(stream);
      ret = mp_isendv_on_stream(v, mh->nblocks, mh->peer, &mh->mem_reg, &mh->req, stream);
      if (ret)
        errorQuda("error in isendv_on_stream %s\n", __FUNCTION__);
    }
    break;
  case MsgHandle_s::TYPE_STRIDED_RECV:
    {
      struct iovec v[mh->nblocks];
      for (int i=0; i<mh->nblocks; ++i) {
        v[i].iov_base = (char*)mh->buffer + i*mh->stride;
        v[i].iov_len = mh->blksize;
      }
      assert(!stream);
      ret = mp_irecvv(v, mh->nblocks, mh->peer, &mh->mem_reg, &mh->req);
      if (ret)
        errorQuda("error in irecvv %s\n", __FUNCTION__);
    }
    break;
  default:
    errorQuda("unsupported msg type %s\n", __FUNCTION__);
  }
}

void comm_start(MsgHandle *mh)
{
  if (mh->gdsync) {
    comm_gdsync_start(mh, 0);
  } else {
    MPI_CHECK( MPI_Start(&(mh->request)) );
  }
}


void comm_start_on_stream(MsgHandle *mh, cudaStream_t stream)
{
  if (mh->gdsync) {
    comm_gdsync_start(mh, stream);
  } else {
    errorQuda("unsupported function %s\n", __FUNCTION__);
  }
}


void comm_wait_on_stream(MsgHandle *mh, cudaStream_t stream)
{
	if (getVerbosity() == QUDA_DEBUG_VERBOSE) printfQuda("[%d] here %s:%d\n", comm_rank(), __FUNCTION__, __LINE__);

  if (mh->gdsync) {
    int ret = mp_wait_on_stream(&mh->req, stream);
    if (ret != SUCCESS)
      errorQuda("error in function %s\n", __FUNCTION__);
  } else {
    errorQuda("unsupported function %s\n", __FUNCTION__);
  }
}


void comm_wait(MsgHandle *mh)
{
	if (getVerbosity() == QUDA_DEBUG_VERBOSE) printfQuda("[%d] here %s:%d\n", comm_rank(), __FUNCTION__, __LINE__);

  if (mh->gdsync) {
    int ret = mp_wait(&mh->req);
    if (ret != SUCCESS)
      errorQuda("error in function %s\n", __FUNCTION__);
  } else {
    MPI_CHECK( MPI_Wait(&(mh->request), MPI_STATUS_IGNORE) );
  }
}


int comm_query(MsgHandle *mh) 
{
  int query;
  if (mh->gdsync) {
    errorQuda("unsupported function %s\n", __FUNCTION__);
  }

  MPI_CHECK( MPI_Test(&(mh->request), &query, MPI_STATUS_IGNORE) );
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
	errorQuda("calling MPI_Abort at %s\n", __FUNCTION__);
  MPI_Abort(MPI_COMM_WORLD, status) ;
}
