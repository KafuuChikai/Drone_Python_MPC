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
        # self.I_xx = 2.5e-3
        # self.I_yy = 2.1e-3
        # self.I_zz = 4.3e-3
        # self.I = np.array([[self.I_xx, .0, .0], [.0, self.I_yy, .0], [.0, .0, self.I_zz]])
        self.m = 1
        self.g = 9.8
        self.TWR_max = 3
        z_axis = np.array([0,0,1])
        k_d = np.array([0.26, 0.28, 0.42])
        k_h = 0.01
        motor_constant = 5.84e-6
        rotor_drag_coefficient = 0.000175
        
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
        z_b_axis = self.q_rot(q, z_axis)
        force_T = T*z_b_axis
        q_inv = self.q_inv(q)
        v_B = self.q_rot(q_inv, v)
        force_drag = -1*k_d*v_B + k_h*(v_B[0]**2 + v_B[1]**2)*np.array([.0, .0, 1])
        force_drag = self.q_rot(q, force_drag)
        real_motor_velocity = ca.sqrt((T+1e-8)/motor_constant)
        velocity_perpendicular_to_rotor_axis = v - v*z_b_axis
        air_drag = -1*real_motor_velocity*rotor_drag_coefficient*velocity_perpendicular_to_rotor_axis
        # paper drag
        f_expression=[v,
                      (force_T + force_drag)/self.m - np.array([.0, .0, self.g]),
                      1/2*self.q_muilty(q, ca.vertcat(0,w))]
        # gazebo drag
        # f_expression=[v,
        #               (force_T + air_drag)/self.m - np.array([.0, .0, self.g]),
        #               1/2*self.q_muilty(q, ca.vertcat(0,w))]

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
        model.name = 'drone_simple_drag'

        # constraint
        constraint.z_max = 100
        constraint.z_min = 0.1
        # constraint.w_max = 1*np.array([np.pi, np.pi, np.pi])
        # constraint.w_min = -constraint.w_max
        constraint.w_max = np.array([3, 3, 0.3])
        constraint.w_min = -constraint.w_max
        # constraint.T_max = 25.7544
        # constraint.T_max = 68.3   #rot 1800, input 0.95
        constraint.T_max = self.m * self.g * self.TWR_max   # rot 1200, input 0.929
        # constraint.T_min = 0.2336     # rot 100
        constraint.T_min = 1

        self.model = model
        self.constraint = constraint
        self.f = f
        
    def q_muilty(self, q1, q2):
        result = ca.vertcat(q1[0]*q2[0] - ca.dot(q1[1:4],q2[1:4]), q1[0]*q2[1:4] + q2[0]*q1[1:4] + ca.cross(q1[1:4],q2[1:4]))
        return result
    
    def q_rot(self, q, axis):
        q_inv = self.q_inv(q)
        cal_axis = ca.vertcat(0,axis)
        new_q = self.q_muilty(self.q_muilty(q,cal_axis), q_inv)
        new_axis = new_q[1:4]
        return new_axis
    
    def q_inv(self, q):
        q_inv = ca.vertcat(q[0],-q[1:4])/ca.norm_2(q)**2
        return q_inv