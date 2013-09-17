#ifndef _COLOR_SPINOR_FIELD_SHIFT_H_
#define _COLOR_SPINOR_FIELD_SHIFT_H_

#include<color_spinor_field.h>

namespace quda{

void shiftColorSpinorField(cudaColorSpinorField &dst, const cudaColorSpinorField &src, const unsigned int parity, const unsigned int dim, const int shift);

};
#endif
