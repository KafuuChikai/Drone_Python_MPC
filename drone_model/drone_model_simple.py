#!/usr/bin/env python
# coding=UTF-8

import numpy as np
import casadi as ca
from acados_template import AcadosModel

class DroneModel(object):
    def __init__(self,):
        #build model
        model = AcadosModel() #  ca.types.SimpleNamespace()
        constraint = ca.types.SimpleNamespace()
        
        # param
        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0, .0], [.0, self.I_yy, .0], [.0, .0, self.I_zz]])
        z_axis = np.array([0,0,1])
        
        # control input
        T = ca.MX.sym('thrust',1)
        w = ca.MX.sym('omega',3)
        controls = ca.vertcat(T, w)
        
        # model states
        p = ca.MX.sym('p',3)    # position
        v = ca.MX.sym('v',3)    # velocity
        q = ca.MX.sym('q',4)    # rotation        
        states = ca.vertcat(p, v, q)        
        
        # function
        f_expression=[v,
                      T*self.q_rot(q, z_axis)/self.m - np.array([.0, .0, self.g]),
                      1/2*self.q_muilty(q, ca.vertcat(0,w))]
        
        f = ca.Function('f', [states, controls], f_expression, ['state', 'control_input'], ['d_p', 'd_v', 'd_q'])

        # acados model
        p_dot = ca.MX.sym('p_dot',3)    # position
        v_dot = ca.MX.sym('v_dot',3)    # velocity
        q_dot = ca.MX.sym('q_dot',4)    # rotation        
        x_dot = ca.vertcat(p_dot, v_dot, q_dot)
        f_impl = x_dot - ca.vcat(f(states, controls))

        model.f_expl_expr = ca.vcat(f(states, controls))
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = 'drone_simple'

        # constraint
        constraint.w_max = 2*np.array([np.pi, np.pi, np.pi])
        constraint.w_min = 2*np.array([-np.pi, -np.pi, -np.pi])
        constraint.T_max = 30
        constraint.T_min = 0

        self.model = model
        self.constraint = constraint
        self.f = f
        
    def q_muilty(self, q1, q2):
        result = ca.vertcat(q1[0]*q2[0] - ca.dot(q1[1:4],q2[1:4]), q1[0]*q2[1:4] + q2[0]*q1[1:4] + ca.cross(q1[1:4],q2[1:4]))
        return result
    
    def q_rot(self, q, axis):
        q_inv = ca.vertcat(q[0],-q[1:4])/ca.norm_2(q)**2
        cal_axis = ca.vertcat(0,axis)
        new_q = self.q_muilty(self.q_muilty(q,cal_axis), q_inv)
        new_axis = new_q[1:4]
        return new_axis