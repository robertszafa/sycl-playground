// as per Kelly (2018)
static inline int nthCleared(int n, int t) {
  int mask = (1 << t) - 1;
  int notMask = ~mask;

  return (n & mask) | ((n & notMask) << 1);
}

/************************ COMPLEX NUMBERS ************************/

//2 component vector to hold the real and imaginary parts of a complex number:
struct cfloat {float x; float y;};

#define cI ((cfloat) (0.0, 1.0))

/*
 * Return Real (Imaginary) component of complex number:
 */
inline float creal(cfloat a) {
    return a.x;
}
inline float cimag(cfloat a) {
    return a.y;
}

// inline float cmod(cfloat a) {
//     return (sqrt(a.x*a.x + a.y*a.y));
// }

// inline float carg(cfloat a) {
//     return atan2(a.y, a.x);
// }

// inline cfloat cadd(cfloat a, cfloat b) {
//     return (cfloat)(a.x + b.x, a.y + b.y);
// }

// inline cfloat cmult(cfloat a, cfloat b) {
//     return (cfloat)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
// }

// inline cfloat cdiv(cfloat a, cfloat b) {
//     return (cfloat)((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
// }

// inline cfloat csqrt(cfloat a) {
//     return (cfloat)(sqrt(cmod(a)) * cos(carg(a)/2),  sqrt(cmod(a)) * sin(carg(a)/2));
// }

typedef struct cpair_ {
    cfloat a;
    cfloat b;
} cpair;

inline cfloat cdot(cpair a, cpair b) { cfloat res = {a.a.x*b.a.x - a.a.y*b.a.y + a.b.x*b.b.x - a.b.y*b.b.y, a.a.y*b.a.x + a.a.x*b.a.y + a.b.x*b.b.y + a.b.y*b.b.x}; return res; }

/************************ END COMPLEX NUMBERS ************************/

typedef struct {
    /**
     * [a, b]
     * [c, d]
     */
    cfloat a;
    cfloat b;
    cfloat c;
    cfloat d;
} GateMatrix;