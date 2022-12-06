#!/usr/bin/env python
# coding=UTF-8

import os
import sys
import shutil
import errno
import timeit

from drone_model_simple import DroneModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

import numpy as np
import scipy.linalg

def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print('Error while removing directory {}'.format(directory))


class DroneOptimizer(object):
    def __init__(self, d_model, d_constraint, t_horizon, n_nodes):
        model = d_model
        self.T = t_horizon
        self.N = n_nodes

        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir = './acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = len(model.p)

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        '''
        Cal type:
        type 1: all ref
        type 2: x, q; u ref
        type 3: x; u ref
        type 4: x ref (?)
        '''

        # cost type 1
        # Q = np.diag([200, 200, 500, 1, 1, 1, 5, 5, 200, 1, 1, 1])
        # R = np.diag([6, 6, 6, 6])
        # Q = np.diag([200, 200, 500, 1, 1, 1, 5, 5, 200])
        # R = np.diag([6, 6, 6, 6])
        # ocp.cost.cost_type = 'LINEAR_LS'
        # ocp.cost.cost_type_e = 'LINEAR_LS'
        # ocp.cost.W = scipy.linalg.block_diag(Q, R)
        # ocp.cost.W_e = Q
        
        # cost type 2
        # Q = np.diag([200, 200, 500, 1, 1, 200])
        # R = np.diag([0.1, 0.1, 0.1, 0.1])
        # ocp.cost.cost_type = 'LINEAR_LS'
        # ocp.cost.cost_type_e = 'LINEAR_LS'
        # ocp.cost.W = scipy.linalg.block_diag(Q, R)
        # ocp.cost.W_e = Q
        
        # cost type 3
        Q = np.diag([200, 200, 500])
        R = np.diag([0, 0, 0, 0])
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q
        
        # q(x,y,z) -> dim: ny-1
        # Vx type 1
        # ocp.cost.Vx = np.zeros(((ny-1), nx))
        # ocp.cost.Vx[:6, :6] = np.eye(6)
        # ocp.cost.Vx[6:9, 7:10] = np.eye(3)
        # ocp.cost.Vx_e = ocp.cost.Vx[:(nx-1), :nx]
        # Vu type 1
        # ocp.cost.Vu = np.zeros(((ny-1), nu))
        # ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        
        # Vx type 2
        # ocp.cost.Vx = np.zeros(((ny-4), nx))
        # ocp.cost.Vx[:3, :3] = np.eye(3)
        # ocp.cost.Vx[3:6, 7:10] = np.eye(3)
        # ocp.cost.Vx_e = ocp.cost.Vx[:(nx-4), :nx]
        # Vu type 2
        # ocp.cost.Vu = np.zeros(((ny-4), nu))
        # ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)

        # Vx type 3
        ocp.cost.Vx = np.zeros(((ny-7), nx))
        ocp.cost.Vx[:3, :3] = np.eye(3)
        ocp.cost.Vx_e = ocp.cost.Vx[:(nx-7), :nx]
        # Vu type 3
        ocp.cost.Vu = np.zeros(((ny-7), nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)

        # set constraints
        # ocp.constraints.lbx = np.array([-6,-6,-6])
        # ocp.constraints.ubx = np.array([6,6,6])
        # ocp.constraints.idxbx = np.array([0,1,2])
        # ocp.constraints.lbx = np.array([-10,-10,-10])
        # ocp.constraints.ubx = np.array([10,10,10])
        # ocp.constraints.idxbx = np.array([3,4,5])
        ocp.constraints.lbu = np.concatenate((np.array([d_constraint.T_min]), d_constraint.w_min))
        ocp.constraints.ubu = np.concatenate((np.array([d_constraint.T_max]), d_constraint.w_max))
        ocp.constraints.idxbu = np.array(range(nu))

        # initial state
        x_init = np.zeros(nx)
        x_init[6] = 1
        ocp.constraints.x0 = x_init
        
        # initial ref type 1
        # u_ref = np.zeros(nu)        
        # x_ref = np.zeros(nx-1)
        # ocp.cost.yref = np.concatenate((x_ref, u_ref))
        # ocp.cost.yref_e = x_ref
        
        # initial ref type 2
        # u_ref = np.zeros(nu)        
        # x_ref = np.zeros(nx-4)
        # ocp.cost.yref = np.concatenate((x_ref, u_ref))
        # ocp.cost.yref_e = x_ref
        
        # initial ref type 3
        u_ref = np.zeros(nu)        
        x_ref = np.zeros(nx-7)
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # solver options
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # explicit Runge-Kutta integrator
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # compile acados ocp
        json_file = os.path.join('./'+model.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def simulation(self, T_max, a_max, v_max, n):
        p0 = self.track_cal(0, a_max, v_max, n)
        v0 = np.array([0, 0, 0])
        q0 = np.array([1, 0, 0, 0])
        x0 = np.concatenate((p0, v0, q0))
        
        Nsim = int(T_max * self.N / self.T)
        Tsim = 0
        
        simX = np.zeros((Nsim+1, self.nx))
        simU = np.zeros((Nsim, self.nu))
        simTrack = np.zeros((Nsim, 3))
        x_current = x0
        simX[0, :] = x0.reshape(1, -1)
        time_record = np.zeros(Nsim)

        # closed loop
        for i in range(Nsim):
            track_ref = self.track_cal(Tsim + self.T, a_max, v_max, n)
            simTrack[i, :] = track_ref
            # yref_between = np.concatenate((track_ref, np.zeros(6), np.zeros(self.nu)))
            yref_between = np.concatenate((track_ref, np.zeros(self.nu)))
            for j in range(self.N):                
                self.solver.set(j, 'yref', yref_between)
            track_ref_N = self.track_cal(Tsim + self.T, a_max, v_max, n)
            # yref_N = np.concatenate((track_ref_N, np.zeros(6)))
            self.solver.set(self.N, 'yref', track_ref_N)
            
            ##  set inertial (stage 0)
            self.solver.set(0, 'lbx', x_current)
            self.solver.set(0, 'ubx', x_current)
            
            # solve
            start = timeit.default_timer()
            status = self.solver.solve()
            if status != 0 :
                raise Exception('acados acados_ocp_solver returned status {}. in closed loop iteration {}.'.format(status, i))
            time_record[i] =  timeit.default_timer() - start
            simU[i, :] = self.solver.get(0, 'u')
            
            # simulate system
            self.integrator.set('x', x_current)
            self.integrator.set('u', simU[i, :])
            status_s = self.integrator.solve()
            if status_s != 0:
                raise Exception('acados integrator returned status {}. in closed loop iteration {}.'.format(status, i))

            # update
            x_current = self.solver.get(1, 'x')
            simX[i+1, :] = x_current
            Tsim = Tsim + self.T/self.N
            print(Tsim)

        print("average estimation time is {}".format(time_record.mean()))
        print("max estimation time is {}".format(time_record.max()))
        print("min estimation time is {}".format(time_record.min()))
        np.savetxt(fname="drone_state.csv", X=simX, fmt="%lf",delimiter=",")
        np.savetxt(fname="drone_control.csv", X=simU, fmt="%lf",delimiter=",")
        np.savetxt(fname="drone_track.csv", X=simTrack, fmt="%lf",delimiter=",")
        
    def track_cal(self, t, a_max, v_max, n):
        r_max = v_max**2/a_max
        k = a_max/v_max
        r_min = r_max/n
        pos_proc = np.array([r_max*np.sin(k*t), r_min*np.cos(k*t), -5])
        return pos_proc

if __name__ == '__main__':
    drone_model = DroneModel()
    opt = DroneOptimizer(d_model=drone_model.model,
                               d_constraint=drone_model.constraint, t_horizon=0.05, n_nodes=20)
    opt.simulation(T_max=3, a_max=20, v_max=10, n=2)