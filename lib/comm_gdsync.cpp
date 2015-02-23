#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <quda_internal.h>
#include <comm_quda.h>


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
};

static int rank = -1;
static int size = -1;
static int gpuid = -1;

static int peer_count = 0;
static int peers[QUDA_MAX_DIM];
static int cart_coords[QUDA_MAX_DIM];
static int cart_rank;

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
    if (true || is_partitioned) {
      // add 2 peers for every partitioned dimension, even when dims[i] == 2
      peer_count += 2;
      npdim += 1;
      pdims[pi] = dims[i];
      period[pi] = 1;
      pi += 1;
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
  }

  MPI_Cart_coords(cartcomm, cart_rank, 2, cart_coords);

  MPI_Barrier(MPI_COMM_WORLD);

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
    mh->gdsync = true;
    mh->type = TYPE_SEND;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    mh->rank = rank;
    mh->peer = comm_peer_from_displacement(displacement);
    mh->mem_reg = mp_register(buffer, nbytes);
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
    mh->gdsync = true;
    mh->type = TYPE_RECV;
    mh->buffer = buffer;
    mh->nbytes = nbytes;
    mh->rank = rank;
    mh->peer = comm_peer_from_displacement(displacement);
    mh->mem_reg = mp_register(buffer, nbytes);
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
    mh->gdsync = true;
    mh->type = TYPE_STRIDED_SEND;
    mh->buffer = buffer;
    mh->size = 0;
    mh->blksize = blksize;
    mh->nblocks = nblocks;
    mh->stride = stride;
    mh->rank = rank;
    mh->peer = comm_peer_from_displacement(displacement);
    assert(blksize <= stride);
    mh->mem_reg = mp_register(buffer, stride*nblocks);
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
    mh->gdsync = true;
    mh->type = TYPE_STRIDED_RECV;
    mh->buffer = buffer;
    mh->size = 0;
    mh->blksize = blksize;
    mh->nblocks = nblocks;
    mh->stride = stride;
    mh->rank = rank;
    mh->peer = comm_peer_from_displacement(displacement);
    assert(blksize <= stride);
    mh->mem_reg = mp_register(buffer, stride*nblocks);
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
  host_free(mh);
}


void comm_gdsync_start(MsgHandle *mh, cudaStream_t stream)
{
  assert(mh->gdsync);

  switch(mh->type) {
  case TYPE_SEND:
    mp_isend_on_stream(mh->buffer, mh->size, mh->peer, mh->mem_reg, &mh->req, stream);
    break;
  case TYPE_RECV:
    mp_irecv(mh->buffer, mh->size, mh->peer, mh->mem_reg, &mh->req);
    break;
  case TYPE_STRIDED_SEND:
    {
      struct iovec v[mh->nblocks];
      for (int i=0; i<mh->nblocks; ++i) {
        v[i].iov_base = mh->buffer + i*stride;
        v[i].iov_len = mh->blksize;
      }
      assert(stream);
      int ret = mp_isendv_on_stream(v, mh->nblocks, mh->peer, mh->mem_reg, &mh->req, stream);
      if (ret)
        errorQuda("error in isendv %s\n", __FUNCTION__);
    }
    break;
  case TYPE_STRIDED_RECV:
    {
      struct iovec v[mh->nblocks];
      for (int i=0; i<mh->nblocks; ++i) {
        v[i].iov_base = mh->buffer + i*stride;
        v[i].iov_len = mh->blksize;
      }
      assert(!stream);
      int ret = mp_irecvv(v, mh->nblocks, mh->peer, mh->mem_reg, &mh->req);
      if (ret)
        errorQuda("error in irecv %s\n", __FUNCTION__);
    }
    break;
  default:
    errorQuda("unsupported msg type %s\n", __FUNCTION__);
  }
}

void comm_start(MsgHandle *mh)
{
  if (mh->gdsync) {
    comm_gdsync_start(mh);
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


void comm_gdsync_wait_on_stream(MsgHandle *mh, cudaStream_t stream)
{
}

void comm_wait_on_stream(MsgHandle *mh, cudaStream_t stream)
{
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
  MPI_Abort(MPI_COMM_WORLD, status) ;
}
