/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) drone_simple_drag_constr_h_e_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};

/* drone_simple_drag_constr_h_e_fun:(i0[10],i1[],i2[],i3[])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real *rr, *ss;
  casadi_real *w0=w+0, w1, w2, w3, w4;
  /* #0: @0 = input[0][2] */
  casadi_copy(arg[0] ? arg[0]+6 : 0, 4, w0);
  /* #1: @1 = @0[0] */
  for (rr=(&w1), ss=w0+0; ss!=w0+1; ss+=1) *rr++ = *ss;
  /* #2: @2 = @0[3] */
  for (rr=(&w2), ss=w0+3; ss!=w0+4; ss+=1) *rr++ = *ss;
  /* #3: @1 = (@1*@2) */
  w1 *= w2;
  /* #4: @2 = @0[1] */
  for (rr=(&w2), ss=w0+1; ss!=w0+2; ss+=1) *rr++ = *ss;
  /* #5: @3 = @0[2] */
  for (rr=(&w3), ss=w0+2; ss!=w0+3; ss+=1) *rr++ = *ss;
  /* #6: @2 = (@2*@3) */
  w2 *= w3;
  /* #7: @1 = (@1+@2) */
  w1 += w2;
  /* #8: @1 = (2.*@1) */
  w1 = (2.* w1 );
  /* #9: @2 = 1 */
  w2 = 1.;
  /* #10: @3 = @0[2] */
  for (rr=(&w3), ss=w0+2; ss!=w0+3; ss+=1) *rr++ = *ss;
  /* #11: @3 = sq(@3) */
  w3 = casadi_sq( w3 );
  /* #12: @4 = @0[3] */
  for (rr=(&w4), ss=w0+3; ss!=w0+4; ss+=1) *rr++ = *ss;
  /* #13: @4 = sq(@4) */
  w4 = casadi_sq( w4 );
  /* #14: @3 = (@3+@4) */
  w3 += w4;
  /* #15: @3 = (2.*@3) */
  w3 = (2.* w3 );
  /* #16: @2 = (@2-@3) */
  w2 -= w3;
  /* #17: @1 = (@1/@2) */
  w1 /= w2;
  /* #18: output[0][0] = @1 */
  if (res[0]) res[0][0] = w1;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_simple_drag_constr_h_e_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_simple_drag_constr_h_e_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_simple_drag_constr_h_e_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_simple_drag_constr_h_e_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_simple_drag_constr_h_e_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_simple_drag_constr_h_e_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_simple_drag_constr_h_e_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_simple_drag_constr_h_e_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_simple_drag_constr_h_e_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int drone_simple_drag_constr_h_e_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real drone_simple_drag_constr_h_e_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_simple_drag_constr_h_e_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_simple_drag_constr_h_e_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_simple_drag_constr_h_e_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_simple_drag_constr_h_e_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_simple_drag_constr_h_e_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 8;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
