#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <typeinfo>

#include <color_spinor_field.h>
#include <blas_quda.h>

#include <string.h>
#include <iostream>
#include <misc_helpers.h>
#include <face_quda.h>
#include <dslash_quda.h>

#ifdef DEVICE_PACK
static const QudaFieldLocation reorder_location_ = QUDA_CUDA_FIELD_LOCATION;
#else
static const QudaFieldLocation reorder_location_ = QUDA_CPU_FIELD_LOCATION;
#endif

int zeroCopy = 0;

namespace quda {

  bool cudaColorSpinorField::initGhostFaceBuffer = false;
  size_t cudaColorSpinorField::ghostFaceBytes = 0;

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : 
    ColorSpinorField(param), alloc(false), init(true), texInit(false),
    ghostTexInit(false), ghost_field_tex{nullptr,nullptr}, bufferMessageHandler(0)
  {
    // this must come before create
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      norm = param.norm;
    }

    create(param.create);

    if  (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_COPY_FIELD_CREATE) {
      errorQuda("not implemented");
    }
  }

  cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) : 
    ColorSpinorField(src), alloc(false), init(true), texInit(false),
    ghostTexInit(false), ghost_field_tex{nullptr,nullptr}, bufferMessageHandler(0)
  {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  // creates a copy of src, any differences defined in param
  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, 
					     const ColorSpinorParam &param) :
    ColorSpinorField(src), alloc(false), init(true), texInit(false),
    ghostTexInit(false), ghost_field_tex{nullptr,nullptr}, bufferMessageHandler(0)
  {
    // can only overide if we are not using a reference or parity special case
    if (param.create != QUDA_REFERENCE_FIELD_CREATE || 
	(param.create == QUDA_REFERENCE_FIELD_CREATE && 
	 src.SiteSubset() == QUDA_FULL_SITE_SUBSET && 
	 param.siteSubset == QUDA_PARITY_SITE_SUBSET && 
	 typeid(src) == typeid(cudaColorSpinorField) ) || 
         (param.create == QUDA_REFERENCE_FIELD_CREATE && (param.is_composite || param.is_component))) {
      reset(param);
    } else {
      errorQuda("Undefined behaviour"); // else silent bug possible?
    }

    // This must be set before create is called
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      if (typeid(src) == typeid(cudaColorSpinorField)) {
	v = (void*)src.V();
	norm = (void*)src.Norm();
      } else {
	errorQuda("Cannot reference a non-cuda field");
      }

      if (composite_descr.is_component && !(src.SiteSubset() == QUDA_FULL_SITE_SUBSET && this->SiteSubset() == QUDA_PARITY_SITE_SUBSET)) 
      {//setup eigenvector form the set
        v    = (void*)((char*)v    + composite_descr.id*bytes);         
        norm = (void*)((char*)norm + composite_descr.id*norm_bytes);         
      }
    }

    create(param.create);

    if (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_COPY_FIELD_CREATE) {
      copySpinorField(src);
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      // do nothing
    } else {
      errorQuda("CreateType %d not implemented", param.create);
    }

  }

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src) 
    : ColorSpinorField(src), alloc(false), init(true), texInit(false),
      ghostTexInit(false), ghost_field_tex{nullptr,nullptr}, bufferMessageHandler(0)
  {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  ColorSpinorField& cudaColorSpinorField::operator=(const ColorSpinorField &src) {
    if (typeid(src) == typeid(cudaColorSpinorField)) {
      *this = (dynamic_cast<const cudaColorSpinorField&>(src));
    } else if (typeid(src) == typeid(cpuColorSpinorField)) {
      *this = (dynamic_cast<const cpuColorSpinorField&>(src));
    } else {
      errorQuda("Unknown input ColorSpinorField %s", typeid(src).name());
    }
    return *this;
  }

  cudaColorSpinorField& cudaColorSpinorField::operator=(const cudaColorSpinorField &src) {
    if (&src != this) {
      // keep current attributes unless unset
      if (!ColorSpinorField::init) { // note this will turn a reference field into a regular field
	destroy();
	destroyComms(); // not sure if this necessary
	ColorSpinorField::operator=(src);
	create(QUDA_COPY_FIELD_CREATE);
      }
      copySpinorField(src);
    }
    return *this;
  }

  cudaColorSpinorField& cudaColorSpinorField::operator=(const cpuColorSpinorField &src) {
    // keep current attributes unless unset
    if (!ColorSpinorField::init) { // note this will turn a reference field into a regular field
      destroy();
      ColorSpinorField::operator=(src);
      create(QUDA_COPY_FIELD_CREATE);
    }
    loadSpinorField(src);
    return *this;
  }

  cudaColorSpinorField::~cudaColorSpinorField() {
    destroyComms();
    destroy();
  }

  void cudaColorSpinorField::create(const QudaFieldCreate create) {

    if (siteSubset == QUDA_FULL_SITE_SUBSET && siteOrder != QUDA_EVEN_ODD_SITE_ORDER) {
      errorQuda("Subset not implemented");
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      v = pool_device_malloc(bytes);
      if (precision == QUDA_HALF_PRECISION) norm = pool_device_malloc(norm_bytes);
      alloc = true;
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      if(composite_descr.is_composite && (create != QUDA_REFERENCE_FIELD_CREATE)) {
	if(composite_descr.dim <= 0) errorQuda("\nComposite size is not defined\n");
	  
        ColorSpinorParam param;
        param.siteSubset = QUDA_FULL_SITE_SUBSET;
        param.nDim = nDim;
        memcpy(param.x, x, nDim*sizeof(int));
        param.create = QUDA_REFERENCE_FIELD_CREATE;
        param.v = v;
        param.norm = norm;
        param.is_composite   = false;
        param.composite_dim  = 0;
        param.is_component = true;

        components.reserve(composite_descr.dim);
        for(int cid = 0; cid < composite_descr.dim; cid++) {
	  param.component_id = cid;
	  components.push_back(new cudaColorSpinorField(*this, param));
        }
      } else {
        // create the associated even and odd subsets
        ColorSpinorParam param;
        param.siteSubset = QUDA_PARITY_SITE_SUBSET;
        param.nDim = nDim;
        memcpy(param.x, x, nDim*sizeof(int));
        param.x[0] /= 2; // set single parity dimensions
        param.create = QUDA_REFERENCE_FIELD_CREATE;
        param.v = v;
        param.norm = norm;
        param.is_composite  = false;
        param.composite_dim = 0;
        param.is_component  = composite_descr.is_component;
        param.component_id  = composite_descr.id;
        even = new cudaColorSpinorField(*this, param);
        odd = new cudaColorSpinorField(*this, param);

        // need this hackery for the moment (need to locate the odd pointers half way into the full field)
        (dynamic_cast<cudaColorSpinorField*>(odd))->v = (void*)((char*)v + bytes/2);
        if (precision == QUDA_HALF_PRECISION) 
	  (dynamic_cast<cudaColorSpinorField*>(odd))->norm = (void*)((char*)norm + norm_bytes/2);

#ifdef USE_TEXTURE_OBJECTS
        dynamic_cast<cudaColorSpinorField*>(even)->destroyTexObject();
        dynamic_cast<cudaColorSpinorField*>(even)->createTexObject();
        dynamic_cast<cudaColorSpinorField*>(odd)->destroyTexObject();
        dynamic_cast<cudaColorSpinorField*>(odd)->createTexObject();
#endif
      }
    } else { //siteSubset == QUDA_PARITY_SITE_SUBSET

      //! setup an object for selected eigenvector (the 1st one as a default):
      if (composite_descr.is_composite && (create != QUDA_REFERENCE_FIELD_CREATE)) 
      {
         if(composite_descr.dim <= 0) errorQuda("\nComposite size is not defined\n");
         //if(bytes > 1811939328) warningQuda("\nCUDA API probably won't be able to create texture object for the eigenvector set... Object size is : %u bytes\n", bytes);
         if (getVerbosity() == QUDA_DEBUG_VERBOSE) printfQuda("\nEigenvector set constructor...\n");
         // create the associated even and odd subsets
         ColorSpinorParam param;
         param.siteSubset = QUDA_PARITY_SITE_SUBSET;
         param.nDim = nDim;
         memcpy(param.x, x, nDim*sizeof(int));
         param.create = QUDA_REFERENCE_FIELD_CREATE;
         param.v = v;
         param.norm = norm;
         param.is_composite   = false;
         param.composite_dim  = 0;
         param.is_component = true;
         //reserve eigvector set
         components.reserve(composite_descr.dim);
         //setup volume, [real_]length and stride for a single eigenvector
         for(int cid = 0; cid < composite_descr.dim; cid++)
         {
            param.component_id = cid;
            components.push_back(new cudaColorSpinorField(*this, param));

#ifdef USE_TEXTURE_OBJECTS //(a lot of texture objects...)
            dynamic_cast<cudaColorSpinorField*>(components[cid])->destroyTexObject();
            dynamic_cast<cudaColorSpinorField*>(components[cid])->createTexObject();
#endif
         }
      }
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (siteSubset != QUDA_FULL_SITE_SUBSET) {
	zeroPad();
      } else if(!composite_descr.is_composite) {
	(dynamic_cast<cudaColorSpinorField*>(even))->zeroPad();
	(dynamic_cast<cudaColorSpinorField*>(odd))->zeroPad();
      } else { //temporary hack for the full spinor field sets, manual zeroPad for each component:
	for(int cid = 0; cid < composite_descr.dim; cid++) {
	  (dynamic_cast<cudaColorSpinorField&>(components[cid]->Even())).zeroPad();
	  (dynamic_cast<cudaColorSpinorField&>(components[cid]->Odd())).zeroPad();
	}
      }
    }

#ifdef USE_TEXTURE_OBJECTS
    if (!composite_descr.is_composite || composite_descr.is_component)
      createTexObject();
#endif
  }

#ifdef USE_TEXTURE_OBJECTS
  void cudaColorSpinorField::createTexObject() {

    if (isNative()) {
      if (texInit) errorQuda("Already bound textures");
      
      // create the texture for the field components
      
      cudaChannelFormatDesc desc;
      memset(&desc, 0, sizeof(cudaChannelFormatDesc));
      if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
      else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2
      
      // staggered and coarse fields in half and single are always two component
      if ( (nSpin == 1 || nSpin == 2) && (precision == QUDA_HALF_PRECISION || precision == QUDA_SINGLE_PRECISION)) {
	desc.x = 8*precision;
	desc.y = 8*precision;
	desc.z = 0;
	desc.w = 0;
      } else { // all others are four component (double2 is spread across int4)
	desc.x = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	desc.y = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	desc.z = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	desc.w = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
      }
      
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = v;
      resDesc.res.linear.desc = desc;
      resDesc.res.linear.sizeInBytes = bytes;
      
      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      if (precision == QUDA_HALF_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
      else texDesc.readMode = cudaReadModeElementType;
      
      cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

      // create the texture for the norm components
      if (precision == QUDA_HALF_PRECISION) {
	cudaChannelFormatDesc desc;
	memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	desc.f = cudaChannelFormatKindFloat;
	desc.x = 8*QUDA_SINGLE_PRECISION; desc.y = 0; desc.z = 0; desc.w = 0;
	
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = norm;
	resDesc.res.linear.desc = desc;
	resDesc.res.linear.sizeInBytes = norm_bytes;
	
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	
	cudaCreateTextureObject(&texNorm, &resDesc, &texDesc, NULL);
      }
      
      texInit = true;

      checkCudaError();
    }
  }

  void cudaColorSpinorField::createGhostTexObject() {
    // create the ghost texture object
    if (isNative() && ghost_bytes) {
      if (ghostTexInit) errorQuda("Already bound ghost texture");

      for (int b=0; b<2; b++) {
	cudaChannelFormatDesc desc;
	memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
	else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2

	// staggered and coarse fields in half and single are always two component
	if ( (nSpin == 1 || nSpin == 2) && (precision == QUDA_HALF_PRECISION || precision == QUDA_SINGLE_PRECISION)) {
	  desc.x = 8*precision;
	  desc.y = 8*precision;
	  desc.z = 0;
	  desc.w = 0;
	} else { // all others are four component (double2 is spread across int4)
	  desc.x = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	  desc.y = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	  desc.z = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	  desc.w = (precision == QUDA_DOUBLE_PRECISION) ? 32 : 8*precision;
	}

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = ghost_field[b];
	resDesc.res.linear.desc = desc;
	resDesc.res.linear.sizeInBytes = ghost_bytes;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	if (precision == QUDA_HALF_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
	else texDesc.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&ghostTex[b], &resDesc, &texDesc, NULL);

	if (precision == QUDA_HALF_PRECISION) {
	  cudaChannelFormatDesc desc;
	  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	  desc.f = cudaChannelFormatKindFloat;
	  desc.x = 8*QUDA_SINGLE_PRECISION; desc.y = 0; desc.z = 0; desc.w = 0;

	  cudaResourceDesc resDesc;
	  memset(&resDesc, 0, sizeof(resDesc));
	  resDesc.resType = cudaResourceTypeLinear;
	  resDesc.res.linear.devPtr = ghost_field[b];
	  resDesc.res.linear.desc = desc;
	  resDesc.res.linear.sizeInBytes = ghost_bytes;

	  cudaTextureDesc texDesc;
	  memset(&texDesc, 0, sizeof(texDesc));
	  texDesc.readMode = cudaReadModeElementType;

	  cudaCreateTextureObject(&ghostTexNorm[b], &resDesc, &texDesc, NULL);
	}

	ghost_field_tex[b] = ghost_field[b];
      } // buffer index

      ghostTexInit = true;

      checkCudaError();
    }
  }

  void cudaColorSpinorField::destroyTexObject() {
    if (isNative() && texInit) {
      cudaDestroyTextureObject(tex);
      if (ghost_bytes) {
	cudaDestroyTextureObject(ghostTex[0]);
	cudaDestroyTextureObject(ghostTex[1]);
      }
      if (precision == QUDA_HALF_PRECISION) {
        cudaDestroyTextureObject(texNorm);
        if (ghost_bytes) {
	  cudaDestroyTextureObject(ghostTexNorm[0]);
	  cudaDestroyTextureObject(ghostTexNorm[1]);
	}
      }
      texInit = false;
    }
  }

  void cudaColorSpinorField::destroyGhostTexObject() {
    if (isNative() && ghostTexInit) {
      cudaDestroyTextureObject(ghostTex[0]);
      cudaDestroyTextureObject(ghostTex[1]);
      if (precision == QUDA_HALF_PRECISION) {
	cudaDestroyTextureObject(ghostTexNorm[0]);
	cudaDestroyTextureObject(ghostTexNorm[1]);
      }
      ghostTexInit = false;
    }
  }
#endif

  void cudaColorSpinorField::destroy() {
    if (alloc) {
      pool_device_free(v);
      if (precision == QUDA_HALF_PRECISION) pool_device_free(norm);
      alloc = false;
    }

    if (composite_descr.is_composite) 
    {
       CompositeColorSpinorField::iterator vec;
       for(vec = components.begin(); vec != components.end(); vec++) delete *vec;
    } 

    if ((siteSubset == QUDA_FULL_SITE_SUBSET && !composite_descr.is_composite) || (siteSubset == QUDA_FULL_SITE_SUBSET && composite_descr.is_component)) {
      delete even;
      delete odd;
    }

#ifdef USE_TEXTURE_OBJECTS
    if (!composite_descr.is_composite || composite_descr.is_component)
      destroyTexObject();
#endif

  }

  // cuda's floating point format, IEEE-754, represents the floating point
  // zero as 4 zero bytes
  void cudaColorSpinorField::zero() {
    cudaMemsetAsync(v, 0, bytes, streams[Nstream-1]);
    if (precision == QUDA_HALF_PRECISION) cudaMemsetAsync(norm, 0, norm_bytes, streams[Nstream-1]);
  }

  void cudaColorSpinorField::zeroPad() {
    size_t pad_bytes = (stride - volume) * precision * fieldOrder;
    int Npad = nColor * nSpin * 2 / fieldOrder;

    if (composite_descr.is_composite && !composite_descr.is_component){//we consider the whole eigenvector set:
      Npad      *= composite_descr.dim;
      pad_bytes /= composite_descr.dim;
    }

    size_t pitch = ((!composite_descr.is_composite || composite_descr.is_component) ? stride : composite_descr.stride)*fieldOrder*precision;
    char   *dst  = (char*)v + ((!composite_descr.is_composite || composite_descr.is_component) ? volume : composite_descr.volume)*fieldOrder*precision;
    if (pad_bytes) cudaMemset2D(dst, pitch, 0, pad_bytes, Npad);

    //for (int i=0; i<Npad; i++) {
    //  if (pad_bytes) cudaMemset((char*)v + (volume + i*stride)*fieldOrder*precision, 0, pad_bytes);
    //}
  }

  void cudaColorSpinorField::copy(const cudaColorSpinorField &src) {
    checkField(*this, src);
    if (this->GammaBasis() != src.GammaBasis()) errorQuda("cannot call this copy with different basis");
    blas::copy(*this, src);
  }

  void cudaColorSpinorField::copySpinorField(const ColorSpinorField &src) {
    
    // src is on the device and is native
    if (typeid(src) == typeid(cudaColorSpinorField) && 
	isNative() && dynamic_cast<const cudaColorSpinorField &>(src).isNative() &&
	this->GammaBasis() == src.GammaBasis()) {
      copy(dynamic_cast<const cudaColorSpinorField&>(src));
    } else if (typeid(src) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION);
    } else if (typeid(src) == typeid(cpuColorSpinorField)) { // src is on the host
      loadSpinorField(src);
    } else {
      errorQuda("Unknown input ColorSpinorField %s", typeid(src).name());
    }
  } 

  void cudaColorSpinorField::loadSpinorField(const ColorSpinorField &src) {

    if (reorder_location_ == QUDA_CPU_FIELD_LOCATION &&typeid(src) == typeid(cpuColorSpinorField)) {
      void *buffer = pool_pinned_malloc(bytes + norm_bytes);
      memset(buffer, 0, bytes+norm_bytes); // FIXME (temporary?) bug fix for padding

      copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, 0, static_cast<char*>(buffer)+bytes, 0);

      qudaMemcpy(v, buffer, bytes, cudaMemcpyHostToDevice);
      qudaMemcpy(norm, static_cast<char*>(buffer)+bytes, norm_bytes, cudaMemcpyHostToDevice);

      pool_pinned_free(buffer);
    } else if (typeid(src) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION);
    } else {
      void *Src=nullptr, *srcNorm=nullptr, *buffer=nullptr;
      if (!zeroCopy) {
	resizeBufferDevice(src.Bytes()+src.NormBytes());
	Src = bufferDevice;
	srcNorm = (char*)bufferDevice + src.Bytes();
	qudaMemcpy(Src, src.V(), src.Bytes(), cudaMemcpyHostToDevice);
	qudaMemcpy(srcNorm, src.Norm(), src.NormBytes(), cudaMemcpyHostToDevice);
      } else {
	buffer = pool_pinned_malloc(src.Bytes()+src.NormBytes());
	memcpy(buffer, src.V(), src.Bytes());
	memcpy(static_cast<char*>(buffer)+src.Bytes(), src.Norm(), src.NormBytes());

	cudaHostGetDevicePointer(&Src, buffer, 0);
	srcNorm = (void*)((char*)Src + src.Bytes());
      }

      cudaMemset(v, 0, bytes); // FIXME (temporary?) bug fix for padding
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, 0, Src, 0, srcNorm);

      if (zeroCopy) pool_pinned_free(buffer);
    }

    return;
  }


  void cudaColorSpinorField::saveSpinorField(ColorSpinorField &dest) const {

    if (reorder_location_ == QUDA_CPU_FIELD_LOCATION &&	typeid(dest) == typeid(cpuColorSpinorField)) {
      void *buffer = pool_pinned_malloc(bytes+norm_bytes);
      qudaMemcpy(buffer, v, bytes, cudaMemcpyDeviceToHost);
      qudaMemcpy(static_cast<char*>(buffer)+bytes, norm, norm_bytes, cudaMemcpyDeviceToHost);

      copyGenericColorSpinor(dest, *this, QUDA_CPU_FIELD_LOCATION, 0, buffer, 0, static_cast<char*>(buffer)+bytes);
      pool_pinned_free(buffer);
    } else if (typeid(dest) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION);
    } else {
      void *dst=nullptr, *dstNorm=nullptr, *buffer=nullptr;
      if (!zeroCopy) {
	resizeBufferDevice(dest.Bytes()+dest.NormBytes());
	dst = bufferDevice;
	dstNorm = (char*)bufferDevice+dest.Bytes();
      } else {
	buffer = pool_pinned_malloc(dest.Bytes()+dest.NormBytes());
	cudaHostGetDevicePointer(&dst, buffer, 0);
	dstNorm = (char*)dst+dest.Bytes();
      }
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION, dst, v, dstNorm, 0);

      if (!zeroCopy) {
	qudaMemcpy(dest.V(), dst, dest.Bytes(), cudaMemcpyDeviceToHost);
	qudaMemcpy(dest.Norm(), dstNorm, dest.NormBytes(), cudaMemcpyDeviceToHost);
      } else {
	memcpy(dest.V(), buffer, dest.Bytes());
	memcpy(dest.Norm(), static_cast<char*>(buffer) + dest.Bytes(), dest.NormBytes());
      }

      if (zeroCopy) pool_pinned_free(buffer);
    }

    return;
  }

  void cudaColorSpinorField::allocateGhostBuffer(int nFace) {

    if (!comm_partitioned()) {
      for (int i=0; i<4; i++) ghost_face_bytes[i] = 0;
      return;
    }

    createGhostZone(nFace);

    // temporary work around until the ghost buffer for fine and
    // coarse grid are merged: this ensures we reset the fine ghost
    // buffer if the coarse grid operator allocates a ghost buffer
    // that is larger than the fine grid operator
    static size_t ghostFaceBytes_ = 0;

    // only allocate if not already allocated or buffer required is bigger than previously
    if ( !initGhostFaceBuffer || ghost_bytes > ghostFaceBytes || ghost_bytes > ghostFaceBytes_) {

      if (initGhostFaceBuffer) {
#ifdef USE_TEXTURE_OBJECTS
	destroyGhostTexObject();
#endif
	if (ghost_bytes) {
	  for (int b=0; b<2; b++) {
	    device_pinned_free(ghost_field[b]);
	    device_free(ghostFaceBuffer[b]);
	    host_free(ghost_pinned_h[b]);
	  }
	}
      }

      if (ghost_bytes > 0) {
	for (int b=0; b<2; ++b) {
	  // gpu receive buffer (use pinned allocator to avoid this being redirected, e.g., by QDPJIT)
	  ghost_field[b] = device_pinned_malloc(ghost_bytes);

	  // gpu send buffset
	  ghostFaceBuffer[b] = device_malloc(ghost_bytes);

	  // pinned buffer used for sending and receiving
	  ghost_pinned_h[b] = pinned_malloc(2*ghost_bytes);

	  // set the matching device-mapper pointer
	  cudaHostGetDevicePointer(&ghost_pinned_d[b], ghost_pinned_h[b], 0);
	}

	initGhostFaceBuffer = true;
	ghostFaceBytes = ghost_bytes;
	ghostFaceBytes_ = ghost_bytes;
      }

      LatticeField::ghost_field_reset = true; // this signals that we must reset the IPC comms
    }

#ifdef USE_TEXTURE_OBJECTS
    // ghost texture is per object
    if (ghost_field_tex[0] != ghost_field[0] || ghost_field_tex[1] != ghost_field[1]) destroyGhostTexObject();
    if (!ghostTexInit) createGhostTexObject();
#endif
  }

  void cudaColorSpinorField::allocateGhostBuffer(void *send_buf[], void *recv_buf[]) const
  {
    int num_faces = 1;
    if (nSpin == 1) num_faces = 3; // staggered

    int spinor_size = 2*nSpin*nColor*precision;

    // resize face only if requested size is larger than previously allocated one
    size_t faceBytes = 0;
    for (int i=0; i<nDimComms; i++) {
      if (comm_dim_partitioned(i)) faceBytes += 2*siteSubset*num_faces*surfaceCB[i]*spinor_size;
    }

    static size_t ghostFaceBytes_ = 0;

    if (!initGhostFaceBuffer || faceBytes > ghostFaceBytes || faceBytes > ghostFaceBytes_) {

      if (initGhostFaceBuffer) {
	for (int b=0; b<2; ++b) device_free(ghostFaceBuffer[b]);
      }

      if (faceBytes > 0) {
	for (int b=0; b<2; ++b) ghostFaceBuffer[b] = device_malloc(faceBytes);
	initGhostFaceBuffer = true;
	ghostFaceBytes = faceBytes;
	ghostFaceBytes_ = faceBytes;
      }

    }

    size_t offset = 0;
    for (int i=0; i<nDimComms; i++) {
      if (comm_dim_partitioned(i)) {
	// use first buffer for recv and second for send
	recv_buf[2*i+0] = static_cast<void*>((static_cast<char*>(ghostFaceBuffer[0]) + offset));
	send_buf[2*i+0] = static_cast<void*>((static_cast<char*>(ghostFaceBuffer[1]) + offset));
	offset += siteSubset*num_faces*surfaceCB[i]*spinor_size;

	recv_buf[2*i+1] = static_cast<void*>((static_cast<char*>(ghostFaceBuffer[0]) + offset));
	send_buf[2*i+1] = static_cast<void*>((static_cast<char*>(ghostFaceBuffer[1]) + offset));
	offset += siteSubset*num_faces*surfaceCB[i]*spinor_size;
      } else {
	recv_buf[2*i+0] = nullptr;
	recv_buf[2*i+1] = nullptr;
	send_buf[2*i+0] = nullptr;
	send_buf[2*i+1] = nullptr;
      }
    }

  }

  void cudaColorSpinorField::freeGhostBuffer(void)
  {
    destroyIPCComms();

    if (!initGhostFaceBuffer) return;
  
    for (int b=0; b<2; b++) {
      // free receive buffer
      if (ghost_field[b]) device_pinned_free(ghost_field[b]);
      ghost_field[b] = nullptr;

      // free send buffer
      if (ghostFaceBuffer[b]) device_free(ghostFaceBuffer[b]);
      ghostFaceBuffer[b] = nullptr;

      // free pinned memory buffers
      if (ghost_pinned_h[b]) host_free(ghost_pinned_h[b]);
      ghost_pinned_h[b] = nullptr;
      ghost_pinned_d[b] = nullptr;
    }
    initGhostFaceBuffer = false;
  }

  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhost(const int nFace, const QudaParity parity, 
                                       const int dim, const QudaDirection dir,
				       const int dagger, cudaStream_t *stream, 
				       void *buffer, bool zero_copy, double a, double b)
  {
#ifdef MULTI_GPU
    int face_num;
    if (dir == QUDA_BACKWARDS) {
      face_num = 0;
    }else if (dir == QUDA_FORWARDS) {
      face_num = 1;
    }else{
      face_num = 2;
    }
    void *packBuffer = buffer ? buffer : ghostFaceBuffer[bufferIndex];
    packFace(packBuffer, *this, zero_copy, nFace, dagger, parity, dim, face_num, *stream, a, b);
#else
    errorQuda("packGhost not built on single-GPU build");
#endif
  }
 
  // send the ghost zone to the host
  void cudaColorSpinorField::sendGhost(void *ghost_spinor, const int nFace, const int dim, 
				       const QudaDirection dir, const int dagger, 
				       cudaStream_t *stream) {

#ifdef MULTI_GPU
    int Nvec = (nSpin == 1 || precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1);  // (spin proj.) degrees of freedom
    int Npad = Nint / Nvec; // number Nvec buffers we have
    
    if (dim !=3 || getKernelPackT() || getTwistPack()) { // use kernels to pack into contiguous buffers then a single cudaMemcpy

      size_t bytes = nFace*Nint*ghostFace[dim]*precision;

      if (precision == QUDA_HALF_PRECISION) bytes += nFace*ghostFace[dim]*sizeof(float);

      void* gpu_buf = 
	(dir == QUDA_BACKWARDS) ? this->backGhostFaceBuffer[bufferIndex][dim] : this->fwdGhostFaceBuffer[bufferIndex][dim];

      cudaMemcpyAsync(ghost_spinor, gpu_buf, bytes, cudaMemcpyDeviceToHost, *stream);

    } else if (this->TwistFlavor() != QUDA_TWIST_NONDEG_DOUBLET) { // do multiple cudaMemcpys

      const int x4 = nDim==5 ? x[4] : 1;
      const int Nt_minus1_offset = (volumeCB - nFace*ghostFace[3])/x4; // N_t -1 = Vh-Vsh

      int offset = 0;
      if (nSpin == 1) {
	offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset;
      } else if (nSpin == 4) {
	// !dagger: send lower components backwards, send upper components forwards
	// dagger: send upper components backwards, send lower components forwards
	bool upper = dagger ? true : false; // Fwd is !Back  
	if (dir == QUDA_FORWARDS) upper = !upper;
	int lower_spin_offset = Npad*stride;
	if (upper) offset = (dir == QUDA_BACKWARDS ? 0 : Nt_minus1_offset);
	else offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? 0 : Nt_minus1_offset);
      }
    
      size_t len = nFace*(ghostFace[3]/x4)*Nvec*precision;
      size_t dpitch = x4*len;
      size_t spitch = stride*Nvec*precision;

      // QUDA Memcpy NPad's worth. 
      //  -- Dest will point to the right beginning PAD. 
      //  -- Each Pad has size Nvec*Vsh Floats. 
      //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next
      for (int s=0; s<x4; s++) { // loop over multiple 4-d volumes (if they exist)
	void *dst = (char*)ghost_spinor + s*len;
	void *src = (char*)v + (offset + s*(volumeCB/x4))*Nvec*precision;
	cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);

	if (precision == QUDA_HALF_PRECISION) {
	  size_t len = nFace*(ghostFace[3]/x4)*sizeof(float);
	  int norm_offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset*sizeof(float);
	  void *dst = (char*)ghost_spinor + nFace*Nint*ghostFace[3]*precision + s*len;
	  void *src = (char*)norm + norm_offset + s*(volumeCB/x4)*sizeof(float);
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream);
	}
      }
    }else{
      int flavorVolume = volume / 2;
      int flavorTFace  = ghostFace[3] / 2;
      int flavor1_Nt_minus1_offset = (flavorVolume - flavorTFace);
      int flavor2_Nt_minus1_offset = (volume - flavorTFace);
      int flavor1_offset = 0;
      int flavor2_offset = 0;
      // !dagger: send lower components backwards, send upper components forwards
      // dagger: send upper components backwards, send lower components forwards
      bool upper = dagger ? true : false; // Fwd is !Back
      if (dir == QUDA_FORWARDS) upper = !upper;
      int lower_spin_offset = Npad*stride;//ndeg tm: stride=2*flavor_volume+pad
      if (upper) {
        flavor1_offset = (dir == QUDA_BACKWARDS ? 0 : flavor1_Nt_minus1_offset);
        flavor2_offset = (dir == QUDA_BACKWARDS ? flavorVolume : flavor2_Nt_minus1_offset);
      }else{
        flavor1_offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? 0 : flavor1_Nt_minus1_offset);
        flavor2_offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? flavorVolume : flavor2_Nt_minus1_offset);
      }

      // QUDA Memcpy NPad's worth.
      //  -- Dest will point to the right beginning PAD.
      //  -- Each Pad has size Nvec*Vsh Floats.
      //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next

      void *dst = (char*)ghost_spinor;
      void *src = (char*)v + flavor1_offset*Nvec*precision;
      size_t len = flavorTFace*Nvec*precision;
      size_t spitch = stride*Nvec*precision;//ndeg tm: stride=2*flavor_volume+pad
      size_t dpitch = 2*len;
      cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);
      dst = (char*)ghost_spinor+len;
      src = (char*)v + flavor2_offset*Nvec*precision;
      cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);

      if (precision == QUDA_HALF_PRECISION) {
        int Nt_minus1_offset = (flavorVolume - flavorTFace);
        int norm_offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset*sizeof(float);
	void *dst = (char*)ghost_spinor + Nint*ghostFace[3]*precision;
	void *src = (char*)norm + norm_offset;
        size_t dpitch = flavorTFace*sizeof(float);
        size_t spitch = flavorVolume*sizeof(float);
	cudaMemcpy2DAsync(dst, dpitch, src, spitch, flavorTFace*sizeof(float), 2, cudaMemcpyDeviceToHost, *stream);
      }
    }
#else
    errorQuda("sendGhost not built on single-GPU build");
#endif

  }



  void cudaColorSpinorField::unpackGhost(const void* ghost_spinor, const int nFace, 
					 const int dim, const QudaDirection dir, 
					 const int dagger, cudaStream_t* stream) 
  {
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1);  // (spin proj.) degrees of freedom

    int len = nFace*ghostFace[dim]*Nint*precision;
    const void *src = ghost_spinor;
  
    int ghost_offset = (dir == QUDA_BACKWARDS) ? ghostOffset[dim][0] : ghostOffset[dim][1];
    void *ghost_dst = (char*)ghost_field[bufferIndex] + precision*ghost_offset;

    if (precision == QUDA_HALF_PRECISION) len += nFace*ghostFace[dim]*sizeof(float);

    cudaMemcpyAsync(ghost_dst, src, len, cudaMemcpyHostToDevice, *stream);
  }




   // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhostExtended(const int nFace, const int R[], const QudaParity parity,
					       const int dim, const QudaDirection dir,
					       const int dagger, cudaStream_t *stream,
					       void *buffer, bool zero_copy)
  {
#ifdef MULTI_GPU
    int face_num;
    if (dir == QUDA_BACKWARDS) {
      face_num = 0;
    }else if (dir == QUDA_FORWARDS) {
      face_num = 1;
    }else{
      face_num = 2;
    }
    void *packBuffer = buffer ? buffer : ghostFaceBuffer[bufferIndex];
    packFaceExtended(packBuffer, *this, zero_copy, nFace, R, dagger, parity, dim, face_num, *stream);
#else
    errorQuda("packGhostExtended not built on single-GPU build");
#endif

  }


  // copy data from host buffer into boundary region of device field
  void cudaColorSpinorField::unpackGhostExtended(const void* ghost_spinor, const int nFace, const QudaParity parity,
                                                 const int dim, const QudaDirection dir, 
                                                 const int dagger, cudaStream_t* stream, bool zero_copy)
  {

     
     
    // First call the regular unpackGhost routine to copy data into the `usual' ghost-zone region 
    // of the data array 
    unpackGhost(ghost_spinor, nFace, dim, dir, dagger, stream);

    // Next step is to copy data from the ghost zone back to the interior region
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1); // (spin proj.) degrees of freedom

    int len = nFace*ghostFace[dim]*Nint;
    int offset = length + ghostOffset[dim][0];
    offset += (dir == QUDA_BACKWARDS) ? 0 : len;

#ifdef MULTI_GPU
    const int face_num = 2;
    const bool unpack = true;
    const int R[4] = {0,0,0,0};
    packFaceExtended(ghostFaceBuffer[bufferIndex], *this, zero_copy, nFace, R, dagger, parity, dim, face_num, *stream, unpack);
#else
    errorQuda("unpackGhostExtended not built on single-GPU build");
#endif
  }


  cudaStream_t *stream;

  void cudaColorSpinorField::createComms(int nFace) {

    allocateGhostBuffer(nFace); // allocate the ghost buffer if not yet allocated

    // ascertain if this instance needs its comms buffers to be updated
    bool comms_reset = ghost_field_reset || // FIXME add send buffer check
      (my_face[0] != ghost_pinned_h[0]) || (my_face[1] != ghost_pinned_h[1]) || // pinned buffers
      (ghost_field_tex[0] != ghost_field[0]) || (ghost_field_tex[1] != ghost_field[1]); // receive buffers

    if (!initComms || comms_reset) {

      // if we are requesting a new number of faces destroy and start over
      destroyComms();

      if (siteSubset != QUDA_PARITY_SITE_SUBSET) errorQuda("Only supports single parity fields");

      int Nint = nColor * nSpin * 2 / (nSpin == 4 ? 2 : 1); // number of internal degrees of freedom

      for (int i=0; i<nDimComms; i++) { // compute size of ghost buffers required
	if (!commDimPartitioned(i)) { ghost_face_bytes[i] = 0; continue; }
	ghost_face_bytes[i] = nFace*ghostFace[i]*Nint*precision;
	if (precision == QUDA_HALF_PRECISION) ghost_face_bytes[i] += nFace*ghostFace[i]*sizeof(float);
      }

      // initialize the ghost pinned buffers
      for (int b=0; b<2; b++) {
	my_face[b] = ghost_pinned_h[b];
	my_face_d[b] = ghost_pinned_d[b];
	from_face[b] = static_cast<char*>(my_face[b]) + ghost_bytes;
	from_face_d[b] = static_cast<char*>(my_face_d[b]) + ghost_bytes;
      }

      // initialize the ghost receive pointers
      for (int i=0; i<nDimComms; ++i) {
	if (commDimPartitioned(i)) {
	  for (int b=0; b<2; b++) {
	    ghost[b][i] = static_cast<char*>(ghost_field[b]) + ghostOffset[i][0]*precision;
	    if (precision == QUDA_HALF_PRECISION)
	      ghostNorm[b][i] = static_cast<char*>(ghost_field[b]) + ghostNormOffset[i][0]*QUDA_SINGLE_PRECISION;
	  }
	}
      }

      // initialize ghost send pointers
      size_t offset = 0;
      for (int i=0; i<nDimComms; i++) {
	if (!commDimPartitioned(i)) continue;

	for (int b=0; b<2; ++b) {
	  backGhostFaceBuffer[b][i] = (void*)(((char*)ghostFaceBuffer[b]) + offset);

	  my_back_face[b][i] = static_cast<char*>(my_face[b]) + offset;
	  from_back_face[b][i] = static_cast<char*>(from_face[b]) + offset;

	  my_back_face_rdma[b][i] = backGhostFaceBuffer[b][i];
	  from_back_face_rdma[b][i] = static_cast<char*>(ghost_field[b]) + ghostOffset[i][0]*precision;
	} // loop over b
	offset += ghost_face_bytes[i];

	for (int b=0; b<2; ++b) {
	  fwdGhostFaceBuffer[b][i] = (void*)(((char*)ghostFaceBuffer[b]) + offset);

	  my_fwd_face[b][i] = static_cast<char*>(my_face[b]) + offset;
	  from_fwd_face[b][i] = static_cast<char*>(from_face[b]) + offset;

	  my_fwd_face_rdma[b][i] = fwdGhostFaceBuffer[b][i];
	  from_fwd_face_rdma[b][i] = static_cast<char*>(ghost_field[b]) + ghostOffset[i][1]*precision;
	} // loop over b
	offset += ghost_face_bytes[i];

      } // loop over dimension

      // initialize the message handlers
      for (int i=0; i<nDimComms; i++) {
	if (!commDimPartitioned(i)) continue;

	for (int b=0; b<2; ++b) {
	  mh_send_fwd[b][i] = comm_declare_send_relative(my_fwd_face[b][i], i, +1, ghost_face_bytes[i]);
	  mh_send_back[b][i] = comm_declare_send_relative(my_back_face[b][i], i, -1, ghost_face_bytes[i]);

	  mh_recv_fwd[b][i] = comm_declare_receive_relative(from_fwd_face[b][i], i, +1, ghost_face_bytes[i]);
	  mh_recv_back[b][i] = comm_declare_receive_relative(from_back_face[b][i], i, -1, ghost_face_bytes[i]);

	  mh_send_rdma_fwd[b][i] = comm_declare_send_relative(my_fwd_face_rdma[b][i], i, +1, ghost_face_bytes[i]);
	  mh_send_rdma_back[b][i] = comm_declare_send_relative(my_back_face_rdma[b][i], i, -1, ghost_face_bytes[i]);

	  mh_recv_rdma_fwd[b][i] = comm_declare_receive_relative(from_fwd_face_rdma[b][i], i, +1, ghost_face_bytes[i]);
	  mh_recv_rdma_back[b][i] = comm_declare_receive_relative(from_back_face_rdma[b][i], i, -1, ghost_face_bytes[i]);
	} // loop over b

      } // loop over dimension
     
      initComms = true;
      checkCudaError();
    }

    if (LatticeField::ghost_field_reset) destroyIPCComms();
    createIPCComms();
  }

  void cudaColorSpinorField::destroyComms()
  {
    if (initComms) {

      for (int b=0; b<2; ++b) {
	for (int i=0; i<nDimComms; i++) {
	  if (commDimPartitioned(i)) {
	    if (mh_recv_fwd[b][i]) comm_free(mh_recv_fwd[b][i]);
	    if (mh_recv_back[b][i]) comm_free(mh_recv_back[b][i]);
	    if (mh_send_fwd[b][i]) comm_free(mh_send_fwd[b][i]);
	    if (mh_send_back[b][i]) comm_free(mh_send_back[b][i]);

	    if (mh_recv_rdma_fwd[b][i]) comm_free(mh_recv_rdma_fwd[b][i]);
	    if (mh_recv_rdma_back[b][i]) comm_free(mh_recv_rdma_back[b][i]);
	    if (mh_send_rdma_fwd[b][i]) comm_free(mh_send_rdma_fwd[b][i]);
	    if (mh_send_rdma_back[b][i]) comm_free(mh_send_rdma_back[b][i]);
	  }
	}
      } // loop over b

      initComms = false;
      checkCudaError();
    }

  }

  void cudaColorSpinorField::streamInit(cudaStream_t *stream_p) {
    stream = stream_p;
  }


  void cudaColorSpinorField::pack(int nFace, int parity, int dagger, cudaStream_t *stream_p, 
				  bool zero_copy, double a, double b) {

    createComms(nFace); // must call this first

    stream = stream_p;
    
    const int dim=-1; // pack all partitioned dimensions
 
    if (zero_copy) {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[0], my_face_d[bufferIndex], true, a, b);
    } else {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[Nstream-1], 0, false, a, b);
    }
  }

  void cudaColorSpinorField::pack(int nFace, int parity, int dagger, int stream_idx, 
				  bool zeroCopyPack, double a, double b) {

    createComms(nFace); // must call this first

    const int dim=-1; // pack all partitioned dimensions
 
    if (zeroCopyPack) {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[stream_idx], my_face_d[bufferIndex], true, a, b);
    } else {
      packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[stream_idx], 0, false, a, b);
    }
  }

  void cudaColorSpinorField::packExtended(const int nFace, const int R[], const int parity, 
                                          const int dagger, const int dim,
                                          cudaStream_t *stream_p, const bool zero_copy) {

    createComms(nFace); // must call this first

    stream = stream_p;
 
    if (zero_copy) {
      packGhostExtended(nFace, R, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[0], my_face_d[bufferIndex], true);
    }else{
      packGhostExtended(nFace, R, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[Nstream-1], 0, false);
    }
  }
                                                      


  void cudaColorSpinorField::gather(int nFace, int dagger, int dir, cudaStream_t* stream_p)
  {
    int dim = dir/2;

    // If stream_p != 0, use pack_stream, else use the stream array
    cudaStream_t *pack_stream = (stream_p) ? stream_p : stream+dir;

    if (dir%2 == 0) {
      // backwards copy to host
      if (comm_peer2peer_enabled(0,dim)) return;

      sendGhost(my_back_face[bufferIndex][dim], nFace, dim, QUDA_BACKWARDS, dagger, pack_stream);
    } else {
      // forwards copy to host
      if (comm_peer2peer_enabled(1,dim)) return;

      sendGhost(my_fwd_face[bufferIndex][dim], nFace, dim, QUDA_FORWARDS, dagger, pack_stream);
    }
  }


  void cudaColorSpinorField::recvStart(int nFace, int dir, int dagger, cudaStream_t* stream_p, bool gdr) {

    int dim = dir/2;
    if (!commDimPartitioned(dim)) return;

    //fprintf(stderr, "%s dir=%d stream_p=%p\n", __func__, dir, stream_p);
    cudaStream_t stream = stream_p ? *stream_p : NULL;

    if (dir%2 == 0) { // sending backwards
      if (comm_peer2peer_enabled(1,dim)) {
        // receive from the processor in the +1 direction
        comm_start(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else if (gdr) {
        // Prepost receive
        comm_start_on_stream(mh_recv_rdma_fwd[bufferIndex][dim], stream);
      } else {
        // Prepost receive
        comm_start_on_stream(mh_recv_fwd[bufferIndex][dim], stream);
      }
    } else { //sending forwards
      // Prepost receive
      if (comm_peer2peer_enabled(0,dim)) {
        comm_start(mh_recv_p2p_back[bufferIndex][dim]);
      } else if (gdr) {
        comm_start_on_stream(mh_recv_rdma_back[bufferIndex][dim], stream);
      } else {
        comm_start_on_stream(mh_recv_back[bufferIndex][dim], stream);
      }
    }
  }


  void cudaColorSpinorField::sendStart(int nFace, int d, int dagger, cudaStream_t* stream_p, bool gdr) {

    int dim = d/2;
    int dir = d%2;
    if (!commDimPartitioned(dim)) return;

    //fprintf(stderr, "%s dir=%d stream_p=%p\n", __func__, dir, stream_p);

    int Nvec = (nSpin == 1 || precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
    int Nint = (nColor * nSpin * 2)/(nSpin == 4 ? 2 : 1); // (spin proj.) degrees of freedom
    int Npad = Nint/Nvec;

    if (!comm_peer2peer_enabled(dir,dim)) {
      cudaStream_t stream = stream_p ? *stream_p : NULL;
      if (dir == 0)
        if (gdr) comm_start_on_stream(mh_send_rdma_back[bufferIndex][dim], stream);
        else comm_start_on_stream(mh_send_back[bufferIndex][dim], stream);
      else
        if (gdr) comm_start_on_stream(mh_send_rdma_fwd[bufferIndex][dim], stream);
        else comm_start_on_stream(mh_send_fwd[bufferIndex][dim], stream);
    } else { // doing peer-to-peer
      cudaStream_t *copy_stream = (stream_p) ? stream_p : stream + d;

      // all goes here
      void* ghost_dst = (dir == 0) ? (void*)((char*)(backGhostSendDest[bufferIndex][dim]) + precision*ghostOffset[dim][1]) :
	(void*)((char*)(fwdGhostSendDest[bufferIndex][dim]) + precision*ghostOffset[dim][0]);
      void *ghost_norm_dst = (dir == 0) ? static_cast<char*>(backGhostSendDest[bufferIndex][dim]) + QUDA_SINGLE_PRECISION*ghostNormOffset[dim][1] :
	static_cast<char*>(fwdGhostSendDest[bufferIndex][dim]) + QUDA_SINGLE_PRECISION*ghostNormOffset[dim][0];

      if (dim != 3 || getKernelPackT()) {

	cudaMemcpyAsync(ghost_dst,
			dir == 0 ? backGhostFaceBuffer[bufferIndex][dim] : fwdGhostFaceBuffer[bufferIndex][dim],
			ghost_face_bytes[dim],
			cudaMemcpyDeviceToDevice,
			*copy_stream); // copy to forward processor

      } else if (this->TwistFlavor() != QUDA_TWIST_NONDEG_DOUBLET) {

	const int x4 = nDim==5 ? x[4] : 1;
	const int Nt_minus_offset = (volumeCB - nFace*ghostFace[3])/x4;

	int offset = 0;
	if (nSpin == 1) {
	  offset = (dir == 0) ? 0 : Nt_minus_offset;
	} else if (nSpin == 4) {
	  // !dagger: send lower components backwards, send upper components forwards
	  // dagger: send upper components backwards, send lower components forwards
	  bool upper = dagger ? true : false;
	  if (dir == 1) upper = !upper;
	  int lower_spin_offset = Npad*stride;
	  if (dir == 0) {
	    offset = upper ? 0 : lower_spin_offset;
	  } else {
	    offset = (upper) ? Nt_minus_offset : lower_spin_offset + Nt_minus_offset;
	  }
	}

	size_t len = nFace*(ghostFace[3]/x4)*Nvec*precision;
	size_t dpitch = x4*len;
	size_t spitch = stride*Nvec*precision;

	for (int s=0; s<x4; s++) {
	  void *dst = (char*)ghost_dst + s*len;
	  void *src = (char*)v + (offset + s*(volumeCB/x4))*Nvec*precision;
	  // start the copy
	  cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

	  if (precision == QUDA_HALF_PRECISION) {
	    size_t len = nFace*(ghostFace[3]/x4)*sizeof(float);
	    int norm_offset = (dir == 0) ? 0 : Nt_minus_offset*sizeof(float);
	    void *dst = (char*)ghost_norm_dst + s*len;
	    void *src = static_cast<char*>(norm) + norm_offset + s*(volumeCB/x4)*sizeof(float);
	    cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToDevice, *copy_stream);
	  }
	}
      } else { // twisted doublet
	int flavorVolume = volume / 2;
	int flavorTFace  = ghostFace[3] / 2;
	int flavor1_Nt_minus1_offset = (flavorVolume - flavorTFace);
	int flavor2_Nt_minus1_offset = (volume - flavorTFace);
	int flavor1_offset = 0;
	int flavor2_offset = 0;
	// !dagger: send lower components backwards, send upper components forwards
	// dagger: send upper components backwards, send lower components forwards
	bool upper = dagger ? true : false; // Fwd is !Back
	if (dir == 1) upper = !upper;
	int lower_spin_offset = Npad*stride;//ndeg tm: stride=2*flavor_volume+pad
	if (upper) {
	  flavor1_offset = (dir == 0 ? 0 : flavor1_Nt_minus1_offset);
	  flavor2_offset = (dir == 0 ? flavorVolume : flavor2_Nt_minus1_offset);
	}else{
	  flavor1_offset = lower_spin_offset + (dir == 0 ? 0 : flavor1_Nt_minus1_offset);
	  flavor2_offset = lower_spin_offset + (dir == 0 ? flavorVolume : flavor2_Nt_minus1_offset);
	}

	// QUDA Memcpy NPad's worth.
	//  -- Dest will point to the right beginning PAD.
	//  -- Each Pad has size Nvec*Vsh Floats.
	//  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next

	void *src = static_cast<char*>(v) + flavor1_offset*Nvec*precision;
	size_t len = flavorTFace*Nvec*precision;
	size_t spitch = stride*Nvec*precision;//ndeg tm: stride=2*flavor_volume+pad
	size_t dpitch = 2*len;
	cudaMemcpy2DAsync(ghost_dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

	src = static_cast<char*>(v) + flavor2_offset*Nvec*precision;
	cudaMemcpy2DAsync(static_cast<char*>(ghost_dst)+len, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

	if (precision == QUDA_HALF_PRECISION) {
	  int norm_offset = (dir == 0) ? 0 : flavor1_Nt_minus1_offset*sizeof(float);
	  void *src = static_cast<char*>(norm) + norm_offset;
	  size_t dpitch = flavorTFace*sizeof(float);
	  size_t spitch = flavorVolume*sizeof(float);
	  cudaMemcpy2DAsync(ghost_norm_dst, dpitch, src, spitch, flavorTFace*sizeof(float), 2, cudaMemcpyDeviceToDevice, *copy_stream);
	}
      }

      if (dir == 0) {
	// record the event
	cudaEventRecord(ipcCopyEvent[bufferIndex][0][dim], *copy_stream);
	// send to the propcessor in the -1 direction
	comm_start(mh_send_p2p_back[bufferIndex][dim]);
      } else {
	cudaEventRecord(ipcCopyEvent[bufferIndex][1][dim], *copy_stream);
	// send to the processor in the +1 direction
	comm_start(mh_send_p2p_fwd[bufferIndex][dim]);
      }
    }
  }

  void cudaColorSpinorField::commsStart(int nFace, int dir, int dagger, cudaStream_t* stream_p, bool gdr) {
    recvStart(nFace, dir, dagger, stream_p, gdr);
    sendStart(nFace, dir, dagger, stream_p, gdr);
  }


  static bool complete_recv_fwd[QUDA_MAX_DIM] = { };
  static bool complete_recv_back[QUDA_MAX_DIM] = { };
  static bool complete_send_fwd[QUDA_MAX_DIM] = { };
  static bool complete_send_back[QUDA_MAX_DIM] = { };

  int cudaColorSpinorField::commsQuery(int nFace, int dir, int dagger, cudaStream_t *stream_p, bool gdr) {

    int dim = dir/2;
    if (!commDimPartitioned(dim)) return 0;

    if (dir%2==0) {

      if (comm_peer2peer_enabled(1,dim)) {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else if (gdr) {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_fwd[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(0,dim)) {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_p2p_back[bufferIndex][dim]);
      } else if (gdr) {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_rdma_back[bufferIndex][dim]);
      } else {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_back[bufferIndex][dim]);
      }

      if (complete_recv_fwd[dim] && complete_send_back[dim]) {
	complete_recv_fwd[dim] = false;
	complete_send_back[dim] = false;
	return 1;
      }

    } else { // dir%2 == 1

      if (comm_peer2peer_enabled(0,dim)) {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_p2p_back[bufferIndex][dim]);
      } else if (gdr) {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_back[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(1,dim)) {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_p2p_fwd[bufferIndex][dim]);
      } else if (gdr) {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_rdma_fwd[bufferIndex][dim]);
      } else {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_fwd[bufferIndex][dim]);
      }

      if (complete_recv_back[dim] && complete_send_fwd[dim]) {
	complete_recv_back[dim] = false;
	complete_send_fwd[dim] = false;
	return 1;
      }

    }

    return 0;
  }

void cudaColorSpinorField::commsWait(int nFace, int dir, int dagger, cudaStream_t *stream_p, bool gdr, unsigned mask) {
    int dim = dir / 2;
    if (!commDimPartitioned(dim)) return;

    //fprintf(stderr, "%s dir=%d stream_p=%p\n", __func__, dir, stream_p);
    cudaStream_t stream = stream_p ? *stream_p : NULL;

    if (dir%2==0) {
      if (mask & wait_recv) {
      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_recv_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][1][dim]);
      } else if (gdr) {
        comm_wait_on_stream(mh_recv_rdma_fwd[bufferIndex][dim], stream);
      } else {
        comm_wait_on_stream(mh_recv_fwd[bufferIndex][dim], stream);
      }
      }
      if (mask & wait_send) {
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_send_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][0][dim]);
      } else if (gdr) {
        comm_wait_on_stream(mh_send_rdma_back[bufferIndex][dim], stream);
      } else {
        comm_wait_on_stream(mh_send_back[bufferIndex][dim], stream);
      }
      }
    } else {
      if (mask & wait_recv) {
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_recv_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][0][dim]);
      } else if (gdr) {
	comm_wait(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
        comm_wait_on_stream(mh_recv_back[bufferIndex][dim], stream);
      }
      }
      if (mask & wait_send) {
      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_send_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][1][dim]);
      } else if (gdr) {
	comm_wait(mh_send_rdma_fwd[bufferIndex][dim]);
      } else {
        comm_wait_on_stream(mh_send_fwd[bufferIndex][dim], stream);
      }
      }
    }

    return;
  }

  void cudaColorSpinorField::scatter(int nFace, int dagger, int dir, cudaStream_t* stream_p)
  {
    int dim = dir/2;
    if (!commDimPartitioned(dim)) return;

    if (dir%2==0) {// receive from forwards
      if (comm_peer2peer_enabled(1,dim)) return;

      unpackGhost(from_fwd_face[bufferIndex][dim], nFace, dim, QUDA_FORWARDS, dagger, stream_p);
    } else { // receive from backwards
      if (comm_peer2peer_enabled(0,dim)) return;

      unpackGhost(from_back_face[bufferIndex][dim], nFace, dim, QUDA_BACKWARDS, dagger, stream_p);
    }
  }



  void cudaColorSpinorField::scatter(int nFace, int dagger, int dir)
  {
    int dim = dir/2;
    if (!commDimPartitioned(dim)) return;

    if (dir%2==0) {// receive from forwards
      if (comm_peer2peer_enabled(1,dim)) return;

      unpackGhost(from_fwd_face[bufferIndex][dim], nFace, dim, QUDA_FORWARDS, dagger, &stream[2*dim+0]);
    } else { // receive from backwards
      if (comm_peer2peer_enabled(0,dim)) return;

      unpackGhost(from_back_face[bufferIndex][dim], nFace, dim, QUDA_BACKWARDS, dagger, &stream[2*dim+1]);
    }
  }

  
  void cudaColorSpinorField::scatterExtended(int nFace, int parity, int dagger, int dir)
  {
    bool zero_copy = false;
    int dim = dir/2;
    if (!commDimPartitioned(dim)) return;
    if (dir%2==0) {// receive from forwards
      unpackGhostExtended(from_fwd_face[bufferIndex][dim], nFace, static_cast<QudaParity>(parity), dim, QUDA_FORWARDS, dagger, &stream[2*dim/*+0*/], zero_copy);
    } else { // receive from backwards
      unpackGhostExtended(from_back_face[bufferIndex][dim], nFace, static_cast<QudaParity>(parity),  dim, QUDA_BACKWARDS, dagger, &stream[2*dim/*+1*/], zero_copy);
    }
  }
 
  void cudaColorSpinorField::exchangeGhost(QudaParity parity, int dagger) const {
    void **send = static_cast<void**>(safe_malloc(nDimComms * 2 * sizeof(void*)));

    // allocate ghost buffer if not yet allocated
    allocateGhostBuffer(send, ghost_fixme);

    genericPackGhost(send, *this, parity, dagger);

    int nFace = (nSpin == 1) ? 3 : 1;
    exchange(ghost_fixme, send, nFace);

    host_free(send);
  }

  std::ostream& operator<<(std::ostream &out, const cudaColorSpinorField &a) {
    out << (const ColorSpinorField&)a;
    out << "v = " << a.v << std::endl;
    out << "norm = " << a.norm << std::endl;
    out << "alloc = " << a.alloc << std::endl;
    out << "init = " << a.init << std::endl;
    return out;
  }

//! for composite fields:
  cudaColorSpinorField& cudaColorSpinorField::Component(const int idx) const {
    
    if (this->IsComposite()) {
      if (idx < this->CompositeDim()) {//setup eigenvector form the set
        return *(dynamic_cast<cudaColorSpinorField*>(components[idx])); 
      }
      else{
        errorQuda("Incorrect component index...");
      }
    }
    errorQuda("Cannot get requested component");
    exit(-1);
  }

//copyCuda currently cannot not work with set of spinor fields..
  void cudaColorSpinorField::CopySubset(cudaColorSpinorField &dst, const int range, const int first_element) const{
#if 0
    if (first_element < 0) errorQuda("\nError: trying to set negative first element.\n");
    if (siteSubset == QUDA_PARITY_SITE_SUBSET && this->EigvId() == -1) {
      if (first_element == 0 && range == this->EigvDim())
      {
        if (range != dst.EigvDim())errorQuda("\nError: eigenvector range to big.\n");
        checkField(dst, *this);
        copyCuda(dst, *this);
      }
      else if ((first_element+range) < this->EigvDim()) 
      {//setup eigenvector subset

        cudaColorSpinorField *eigv_subset;

        ColorSpinorParam param;

        param.nColor = nColor;
        param.nSpin = nSpin;
        param.twistFlavor = twistFlavor;
        param.precision = precision;
        param.nDim = nDim;
        param.pad = pad;
        param.siteSubset = siteSubset;
        param.siteOrder = siteOrder;
        param.fieldOrder = fieldOrder;
        param.gammaBasis = gammaBasis;
        memcpy(param.x, x, nDim*sizeof(int));
        param.create = QUDA_REFERENCE_FIELD_CREATE;
 
        param.eigv_dim  = range;
        param.eigv_id   = -1;
        param.v = (void*)((char*)v + first_element*eigv_bytes);
        param.norm = (void*)((char*)norm + first_element*eigv_norm_bytes);

        eigv_subset = new cudaColorSpinorField(param);

        //Not really needed:
        eigv_subset->eigenvectors.reserve(param.eigv_dim);
        for (int id = first_element; id < (first_element+range); id++)
        {
            param.eigv_id = id;
            eigv_subset->eigenvectors.push_back(new cudaColorSpinorField(*this, param));
        }
        checkField(dst, *eigv_subset);
        copyCuda(dst, *eigv_subset);

        delete eigv_subset;
      } else {
        errorQuda("Incorrect eigenvector dimension...");
      }
    } else{
      errorQuda("Eigenvector must be a parity spinor");
      exit(-1);
    }
#endif
  }

  void cudaColorSpinorField::getTexObjectInfo() const
  {
#ifdef USE_TEXTURE_OBJECTS
    printfQuda("\nPrint texture info for the field:\n");
    std::cout << *this;
    cudaResourceDesc resDesc;
    //memset(&resDesc, 0, sizeof(resDesc));
    cudaGetTextureObjectResourceDesc(&resDesc, this->Tex());
    printfQuda("\nDevice pointer: %p\n", resDesc.res.linear.devPtr);
    printfQuda("\nVolume (in bytes): %lu\n", resDesc.res.linear.sizeInBytes);
    if (resDesc.resType == cudaResourceTypeLinear) printfQuda("\nResource type: linear \n");
#endif
  }

  void cudaColorSpinorField::Source(const QudaSourceType sourceType, const int st, const int s, const int c) {
    ColorSpinorParam param(*this);
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.create = QUDA_NULL_FIELD_CREATE;

    cpuColorSpinorField tmp(param);
    tmp.Source(sourceType, st, s, c);
    *this = tmp;
  }

  void cudaColorSpinorField::PrintVector(unsigned int i) {
    ColorSpinorParam param(*this);
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.create = QUDA_NULL_FIELD_CREATE;

    cpuColorSpinorField tmp(param);
    tmp = *this;
    tmp.PrintVector(i);
  }


} // namespace quda
