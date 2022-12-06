/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_drone_H_
#define ACADOS_SOLVER_drone_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define DRONE_NX     13
#define DRONE_NZ     0
#define DRONE_NU     4
#define DRONE_NP     0
#define DRONE_NBX    3
#define DRONE_NBX0   13
#define DRONE_NBU    4
#define DRONE_NSBX   0
#define DRONE_NSBU   0
#define DRONE_NSH    0
#define DRONE_NSG    0
#define DRONE_NSPHI  0
#define DRONE_NSHN   0
#define DRONE_NSGN   0
#define DRONE_NSPHIN 0
#define DRONE_NSBXN  0
#define DRONE_NS     0
#define DRONE_NSN    0
#define DRONE_NG     0
#define DRONE_NBXN   0
#define DRONE_NGN    0
#define DRONE_NY0    16
#define DRONE_NY     16
#define DRONE_NYN    12
#define DRONE_N      20
#define DRONE_NH     0
#define DRONE_NPHI   0
#define DRONE_NHN    0
#define DRONE_NPHIN  0
#define DRONE_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct drone_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;




    // cost






    // constraints




} drone_solver_capsule;

ACADOS_SYMBOL_EXPORT drone_solver_capsule * drone_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int drone_acados_free_capsule(drone_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int drone_acados_create(drone_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int drone_acados_reset(drone_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of drone_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int drone_acados_create_with_discretization(drone_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int drone_acados_update_time_steps(drone_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int drone_acados_update_qp_solver_cond_N(drone_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int drone_acados_update_params(drone_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int drone_acados_update_params_sparse(drone_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int drone_acados_solve(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int drone_acados_free(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void drone_acados_print_stats(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int drone_acados_custom_update(drone_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *drone_acados_get_nlp_in(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *drone_acados_get_nlp_out(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *drone_acados_get_sens_out(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *drone_acados_get_nlp_solver(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *drone_acados_get_nlp_config(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *drone_acados_get_nlp_opts(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *drone_acados_get_nlp_dims(drone_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *drone_acados_get_nlp_plan(drone_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_drone_H_
