#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

namespace quda {

  template <typename Float>
  ColorSpinorFieldOrder<Float>* createOrder(const cpuColorSpinorField &a) {
    ColorSpinorFieldOrder<Float>* ptr=0;
    if (a.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) 
      ptr = new SpaceSpinColorOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    else if (a.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) 
      ptr = new SpaceColorSpinOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    else if (a.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
      ptr = new QOPDomainWallOrder<Float>(const_cast<cpuColorSpinorField&>(a));
    else
      errorQuda("Order %d not supported in cpuColorSpinorField", a.FieldOrder());
    return ptr;
  }

  // Applies the grid restriction operator (fine to coarse)
  template <class CoarseSpinor, class FineSpinor>
  void restrict(CoarseSpinor &out, const FineSpinor &in, const int* geo_map, const int* spin_map) {

    // We need to zero all elements first, since this is a reduction operation
    for (int x=0; x<in.Volume(); x++) {
      for (int s=0; s<in.Nspin(); s++) {
	for (int c=0; c<in.Ncolor(); c++) {
	  out(geo_map[x], spin_map[s], c) = 0.0;
	}
      }
    }

    for (int x=0; x<in.Volume(); x++) {
      for (int s=0; s<in.Nspin(); s++) {
	for (int c=0; c<in.Ncolor(); c++) {
	  out(geo_map[x], spin_map[s], c) += in(x, s, c);
	}
      }
    }

  }

  /*
    Rotates from the fine-color basis into the coarse-color basis.
  */
  template <class CoarseColor, class FineColor, class Rotator>
  void rotateCoarseColor(CoarseColor &out, const FineColor &in, const Rotator &V) {

    for(int x=0; x<in.Volume(); x++) {

      for (int s=0; s<out.Nspin(); s++) for (int i=0; i<out.Ncolor(); i++) out(x, s, i) = 0.0;

      for (int i=0; i<out.Ncolor(); i++) {
	for (int s=0; s<out.Nspin(); s++) {
	  for (int j=0; j<in.Ncolor(); j++) {
	    out(x, s, j) += std::conj(V(x, i, s*in.Ncolor() + j)) * in(x, s, i);
	  }
	}
      }
    }

  }

  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		ColorSpinorField &tmp, const int *geo_map, const int *spin_map) {
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      ColorSpinorFieldOrder<double> *outOrder = createOrder<double>(out);
      ColorSpinorFieldOrder<double> *inOrder = createOrder<double>(in);
      ColorSpinorFieldOrder<double> *vOrder = createOrder<double>(v);
      ColorSpinorFieldOrder<double> *tmpOrder = createOrder<double>(tmp);
      rotateCoarseColor(*tmpOrder, *inOrder, *vOrder);
      restrict(*outOrder, *tmpOrder, geo_map, spin_map);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    } else {
      ColorSpinorFieldOrder<float> *outOrder = createOrder<float>(out);
      ColorSpinorFieldOrder<float> *inOrder = createOrder<float>(in);
      ColorSpinorFieldOrder<float> *vOrder = createOrder<float>(v);
      ColorSpinorFieldOrder<float> *tmpOrder = createOrder<float>(tmp);
      rotateCoarseColor(*tmpOrder, *inOrder, *vOrder);
      restrict(*outOrder, *tmpOrder, geo_map, spin_map);
      delete outOrder;
      delete inOrder;
      delete vOrder;
      delete tmpOrder;
    }
  }

} // namespace quda
