#ifndef __STAGGERED_OPROD_H__
#define __STAGGERED_OPROD_H__

#include <face_quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>

namespace quda {

  void createStaggeredOprodEvents();
  void destroyStaggeredOprodEvents();

  void computeStagggeredOprod(cudaGaugeField& out, const cudaColorSpinorField& in,
                              FaceBuffer& facebuffer, const unsigned int parity, const unsigned int displacement);

} // namespace quda
  

#endif
