enum IndexType {
  EVEN_X = 0,
  EVEN_Y = 1,
  EVEN_Z = 2,
  EVEN_T = 3
};

template <IndexType idxType>
static __device__ __forceinline__ void coordsFromIndex(int& idx, int c[4],  
						       const unsigned int cb_idx, const unsigned int parity, const unsigned int X[4])
{
  const unsigned int &LX = X[0];
  const unsigned int &LY = X[1];
  const unsigned int &LZ = X[2];
  const unsigned int XYZ = X[2]*X[1]*X[0];
  const unsigned int XY = X[1]*X[0];

  idx = 2*cb_idx;

  int x, y, z, t;

  if (idxType == EVEN_X /*!(LX & 1)*/) { // X even
    //   t = idx / XYZ;
    //   z = (idx / XY) % Z;
    //   y = (idx / X) % Y;
    //   idx += (parity + t + z + y) & 1;
    //   x = idx % X;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = idx / LX;
    x = idx - aux1 * LX;
    int aux2 = aux1 / LY;
    y = aux1 - aux2 * LY;
    t = aux2 / LZ;
    z = aux2 - t * LZ;
    aux1 = (parity + t + z + y) & 1;
    x += aux1;
    idx += aux1;
  } else if (idxType == EVEN_Y /*!(LY & 1)*/) { // Y even
    t = idx / XYZ;
    z = (idx / XY) % LZ;
    idx += (parity + t + z) & 1;
    y = (idx / LX) % LY;
    x = idx % LX;
  } else if (idxType == EVEN_Z /*!(LZ & 1)*/) { // Z even
    t = idx / XYZ;
    idx += (parity + t) & 1;
    z = (idx / XY) % LZ;
    y = (idx / LX) % LY;
    x = idx % LX;
  } else {
    idx += parity;
    t = idx / XYZ;
    z = (idx / XY) % LZ;
    y = (idx / LX) % LY;
    x = idx % LX;
  }

  c[0] = x;
  c[1] = y;
  c[2] = z;
  c[3] = t;
}


