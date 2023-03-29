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
        self.TWR_max = 5
        z_axis = np.array([0,0,1])
        # k_d = np.array([0.26, 0.28, 0.42])
        # k_h = 0.01
        # k_d = np.array([0.5465, 0.4592, 0.0491])
        # k_h = -0.0014
        # k_d = np.array([0.57, 0.52, 0.068])
        # k_h = 0.013
        k_d = np.array([0.5173, 0.5305, -0.0893])
        k_h = 0.0386
        k_h2 = -0.0768
        # k_d = np.array([0.5173, 0.5305, -0.1427])
        # k_h = -0.0527
        # k_h2 = -0.1149
        motor_constant = 5.84e-6
        rotor_drag_coefficient = 0.000175
        kw = 20  # rate delay
        kT = 10  # thrust delay
        
        # control input(delay)
        T_in = ca.MX.sym('thrust',1)
        w_in = ca.MX.sym('omega',3)
        controls = ca.vertcat(T_in, w_in)
        
        # model states
        p = ca.MX.sym('p',3)    # position
        v = ca.MX.sym('v',3)    # velocity
        q = ca.MX.sym('q',4)    # rotation 
        w = ca.MX.sym('w',3)    # body rate
        T = ca.MX.sym('T',1)    # thrust
        # states = ca.vertcat(p, v, q, w, T)
        states = ca.vertcat(p, v, q, w)        

        # constraint function
        phi_constraint = 2*(q[0]*q[1] + q[2]*q[3]) / (1-2*(q[1]*q[1] + q[2]*q[2]))
        theta_constraint = 2*(q[0]*q[2] - q[1]*q[3])
        psi_constraint = 2*(q[0]*q[3] + q[1]*q[2])/ (1-2*(q[2]*q[2] + q[3]*q[3]))
        attitude_constraint = ca.vertcat(phi_constraint, theta_constraint, psi_constraint)

        # function
        z_b_axis = self.q_rot(q, z_axis)
        # force_T = T*z_b_axis
        force_T = T_in*z_b_axis
        q_inv = self.q_inv(q)
        v_B = self.q_rot(q_inv, v)
        force_drag = -1*k_d*v_B + (k_h*(v_B[0]*v_B[0] + v_B[1]*v_B[1])+k_h2*v_B[2]*v_B[2])*np.array([.0, .0, 1])
        force_drag = self.q_rot(q, force_drag)
        # force_drag = 0  # test no drag
        real_motor_velocity = ca.sqrt((T+1e-8)/motor_constant)
        velocity_perpendicular_to_rotor_axis = v - v*z_b_axis
        air_drag = -1*real_motor_velocity*rotor_drag_coefficient*velocity_perpendicular_to_rotor_axis
        # paper drag
        # f_expression=[v,
        #               (force_T + force_drag)/self.m - np.array([.0, .0, self.g]),
        #               1/2*self.q_muilty(q, ca.vertcat(0,w)),
        #               kw*(w_in - w),
        #               kT*(T_in - T)]
        f_expression=[v,
                      (force_T + force_drag)/self.m - np.array([.0, .0, self.g]),
                      1/2*self.q_muilty(q, ca.vertcat(0,w)),
                      kw*(w_in - w)]

        # gazebo drag
        # f_expression=[v,
        #               (force_T + air_drag)/self.m - np.array([.0, .0, self.g]),
        #               1/2*self.q_muilty(q, ca.vertcat(0,w))]

        # f = ca.Function('f', [states, controls], f_expression, ['state', 'control_input'], ['d_p', 'd_v', 'd_q', 'd_w', 'd_T'])
        f = ca.Function('f', [states, controls], f_expression, ['state', 'control_input'], ['d_p', 'd_v', 'd_q', 'd_w'])

        # acados model
        p_dot = ca.MX.sym('p_dot',3)    # position
        v_dot = ca.MX.sym('v_dot',3)    # velocity
        q_dot = ca.MX.sym('q_dot',4)    # rotation
        w_dot = ca.MX.sym('w_dot',3)    # body rate
        T_dot = ca.MX.sym('T_dot',1)    # thrust        
        # x_dot = ca.vertcat(p_dot, v_dot, q_dot, w_dot, T_dot)
        x_dot = ca.vertcat(p_dot, v_dot, q_dot, w_dot)
        f_impl = x_dot - ca.vcat(f(states, controls))

        model.f_expl_expr = ca.vcat(f(states, controls))
        model.f_impl_expr = f_impl
        # model.con_h_expr = psi_constraint
        # model.con_h_expr_e = psi_constraint
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
        constraint.w_max = np.array([5, 5, 0.3])
        constraint.w_min = -constraint.w_max
        # constraint.T_max = 25.7544
        # constraint.T_max = 68.3   #rot 1800, input 0.95
        constraint.T_max = self.m * self.g * self.TWR_max   # rot 1200, input 0.929
        # constraint.T_min = 0.2336     # rot 100
        constraint.T_min = 4
        # constraint.psi_max = np.tan(np.pi / 2)
        # constraint.psi_min = -constraint.psi_max

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