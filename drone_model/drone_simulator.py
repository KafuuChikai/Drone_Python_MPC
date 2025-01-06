import math
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sin, cos, tan

from drone_model import DroneModel

import casadi as ca

class DroneControlSim:
    def __init__(self):
        self.model = DroneModel()
        
        self.sim_time = 50
        self.sim_step = 0.002
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 12))
        self.time = np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3))
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3))
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3))
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3))
        self.pointer = 0

        #Debug
        self.M = np.zeros((int(self.sim_time/self.sim_step), 3))
        self.T = np.zeros((int(self.sim_time/self.sim_step), 1))

        self.xei = 0
        self.xelast = 0

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array(
            [[self.I_xx, .0, .0], [.0, self.I_yy, .0], [.0, .0, self.I_zz]])

    def run(self):
        for self.pointer in range(self.drone_states.shape[0]-1):
            self.time[self.pointer] = self.pointer * self.sim_step
            thrust_cmd = 0
            M = np.zeros((3,))

            #Velocity Control
            self.velocity_cmd[self.pointer] = self.position_controller(self.position_cmd[self.pointer])

            #Attitude & Thrust Control
            [thrust_cmd, phi, theta] = self.velocity_controller(self.velocity_cmd[self.pointer])
            self.attitude_cmd[self.pointer, 0] = phi
            self.attitude_cmd[self.pointer, 1] = theta

            #Rate Control
            self.rate_cmd[self.pointer] = self.attitude_controller(self.attitude_cmd[self.pointer])

            #Moment Control
            M = self.rate_controller(self.rate_cmd[self.pointer])

            #real: less than 10 N
            # if thrust_cmd < -10:
            #     thrust_cmd = -10

            #Update
            # dx = self.drone_dynamics(thrust_cmd, M)
            # self.drone_states[self.pointer + 1] = self.drone_states[self.pointer] + dx * self.sim_step
            control_cmd = np.concatenate((np.array([thrust_cmd]),M))
            state_now = self.drone_states[self.pointer]
            
            euler = state_now[6:9]
            q = self.euler2quatern(euler)
            
            state_now_q = np.concatenate((state_now[0:6], q, state_now[9:12]))
            dx_q = np.array(ca.vcat(self.model.f(state_now_q, control_cmd)))            
            dx_q = dx_q.reshape(13)
            
            state_next_q = state_now_q + dx_q * self.sim_step
            euler = self.quatern2euler(state_next_q[6:10]) 

            self.drone_states[self.pointer + 1] = np.concatenate((state_next_q[0:6], euler, state_next_q[10:13]))
            # print(self.drone_states[self.pointer + 1])
            #Debug
            self.M[self.pointer] = M
            self.T[self.pointer] = thrust_cmd

        self.time[-1] = self.sim_time

    def drone_dynamics(self, T, M):
        # Input:
        # T: float Thrust
        # M: np.array (3,)  Moments in three axes
        # Output: np.array (12,) the derivative (dx) of the drone

        x = self.drone_states[self.pointer, 0]
        y = self.drone_states[self.pointer, 1]
        z = self.drone_states[self.pointer, 2]
        vx = self.drone_states[self.pointer, 3]
        vy = self.drone_states[self.pointer, 4]
        vz = self.drone_states[self.pointer, 5]
        phi = self.drone_states[self.pointer, 6]
        theta = self.drone_states[self.pointer, 7]
        psi = self.drone_states[self.pointer, 8]
        p = self.drone_states[self.pointer, 9]
        q = self.drone_states[self.pointer, 10]
        r = self.drone_states[self.pointer, 11]

        R_d_angle = np.array([[1, tan(theta)*sin(phi), tan(theta)*cos(phi)],
                             [0, cos(phi), -sin(phi)],
                             [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]])

        R_E_B = np.array([[cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta)],
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), sin(phi) *
                           sin(theta)*sin(psi)+cos(phi)*cos(psi), sin(phi)*cos(theta)],
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi), cos(phi)*cos(theta)]])

        d_position = np.array([vx, vy, vz])
        d_velocity = np.array([.0, .0, self.g]) + \
            R_E_B.transpose()@np.array([.0, .0, T])/self.m
        d_angle = R_d_angle@np.array([p, q, r])
        d_q = np.linalg.inv(
            self.I)@(M-np.cross(np.array([p, q, r]), self.I@np.array([p, q, r])))

        dx = np.concatenate((d_position, d_velocity, d_angle, d_q))

        return dx

    def rate_controller(self, cmd):
        # Input: cmd np.array (3,) rate commands
        # Output: M np.array (3,) moments
        k_p = 1
        M = k_p*(cmd - self.drone_states[self.pointer, [9, 10, 11]])
        return M

    def attitude_controller(self, cmd):
        # Input: cmd np.array (3,) attitude commands
        # Output: M np.array (3,) rate commands
        k_p = 1
        phi = self.drone_states[self.pointer, 6]
        theta = self.drone_states[self.pointer, 7]
        R_d_angle = np.array([[1, tan(theta)*sin(phi), tan(theta)*cos(phi)],
                             [0, cos(phi), -sin(phi)],
                             [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]])
        rate = k_p*np.linalg.inv(R_d_angle)@(cmd - self.drone_states[self.pointer, [6, 7, 8]])
        return rate

    def velocity_controller(self, cmd):
        # Input: cmd np.array (3,) velocity commands
        # Output: M np.array (2,) phi and theta commands and thrust cmd
        k_p = 1

        phi = self.drone_states[self.pointer, 6]
        theta = self.drone_states[self.pointer, 7]
        psi = self.drone_states[self.pointer, 8]


        R_E_B = np.array([[cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta)],
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), sin(phi) *
                           sin(theta)*sin(psi)+cos(phi)*cos(psi), sin(phi)*cos(theta)],
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi), cos(phi)*cos(theta)]])

        # output = R_E_B@(k_p*(cmd - self.drone_states[self.pointer, [3, 4, 5]]) - np.array([.0, .0, self.g]))
        # output2=k_p*(cmd - self.drone_states[self.pointer, [3, 4, 5]]) - np.array([.0, .0, self.g])
        # thrust = output[2]*self.m
        # phi = k_p*math.asin((output2[1]*cos(psi) - output2[0]*sin(psi))/np.linalg.norm(output2))
        # theta = -1*k_p*math.asin((output2[0]*cos(psi) + output2[1]*sin(psi))/np.linalg.norm(output2))

        output = k_p*(cmd - self.drone_states[self.pointer, [3, 4, 5]]) - np.array([.0, .0, self.g])
        thrust = output[2]*self.m
        phi = k_p*(output[1]*cos(psi) - output[0]*sin(psi))/10
        theta = -1*k_p*(output[0]*cos(psi) + output[1]*sin(psi))/10
        return thrust, phi, theta

    def position_controller(self, cmd):
        # Input: cmd np.array (3,) position commands
        # Output: M np.array (3,) velocity commands
        k_p = 0.5
        k_i = 0
        k_d = 0
        er = cmd - self.drone_states[self.pointer, [0, 1, 2]]
        self.xei = self.xei + self.sim_step*er
        if self.pointer == 0:
            d = 0
        else:
            d = (er - self.xelast)/self.sim_step
        v = k_p*er + k_i*self.xei + k_d*d
        self.xelast = er
        return v

    def plot_states(self):
        fig1, ax1 = plt.subplots(6, 3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0, 0].plot(self.time, self.drone_states[:, 0], label='real')
        ax1[0, 0].plot(self.time, self.position_cmd[:, 0], label='cmd')
        ax1[0, 0].set_ylabel('x[m]')
        ax1[0, 1].plot(self.time, self.drone_states[:, 1])
        ax1[0, 1].plot(self.time, self.position_cmd[:, 1])
        ax1[0, 1].set_ylabel('y[m]')
        ax1[0, 2].plot(self.time, -self.drone_states[:, 2])
        ax1[0, 2].plot(self.time, -self.position_cmd[:, 2])
        ax1[0, 2].set_ylabel('z[m]')
        ax1[0, 0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1, 0].plot(self.time, self.drone_states[:, 3])
        ax1[1, 0].plot(self.time, self.velocity_cmd[:, 0])
        ax1[1, 0].set_ylabel('vx[m/s]')
        ax1[1, 1].plot(self.time, self.drone_states[:, 4])
        ax1[1, 1].plot(self.time, self.velocity_cmd[:, 1])
        ax1[1, 1].set_ylabel('vy[m/s]')
        ax1[1, 2].plot(self.time, -self.drone_states[:, 5])
        ax1[1, 2].plot(self.time, -self.velocity_cmd[:, 2])
        ax1[1, 2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2, 0].plot(self.time, self.drone_states[:, 6])
        ax1[2, 0].plot(self.time, self.attitude_cmd[:, 0])
        ax1[2, 0].set_ylabel('phi[rad]')
        ax1[2, 1].plot(self.time, self.drone_states[:, 7])
        ax1[2, 1].plot(self.time, self.attitude_cmd[:, 1])
        ax1[2, 1].set_ylabel('theta[rad]')
        ax1[2, 2].plot(self.time, -self.drone_states[:, 8])
        ax1[2, 2].plot(self.time, -self.attitude_cmd[:, 2])
        ax1[2, 2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3, 0].plot(self.time, self.drone_states[:, 9])
        ax1[3, 0].plot(self.time, self.rate_cmd[:, 0])
        ax1[3, 0].set_ylabel('p[rad/s]')
        ax1[3, 1].plot(self.time, self.drone_states[:, 10])
        ax1[3, 1].plot(self.time, self.rate_cmd[:, 1])
        ax1[3, 0].set_ylabel('q[rad/s]')
        ax1[3, 2].plot(self.time, -self.drone_states[:, 11])
        ax1[3, 2].plot(self.time, -self.rate_cmd[:, 2])
        ax1[3, 0].set_ylabel('r[rad/s]')

        #Debug
        self.M[-1] = self.M[-2]
        ax1[4, 0].plot(self.time, self.M[:, 0])
        ax1[4, 0].set_ylabel('Mx')
        ax1[4, 1].plot(self.time, self.M[:, 1])
        ax1[4, 0].set_ylabel('My')
        ax1[4, 2].plot(self.time, -self.M[:, 2])
        ax1[4, 0].set_ylabel('Mz')

        self.T[-1] = self.T[-2]
        ax1[5, 0].plot(self.time, -self.T[:])
        ax1[5, 0].set_ylabel('T')
        
    def quatern2euler(self, q):
        R11 = -1*(2*q[1]**2 - 1 + 2*q[2]**2)
        R21 = 2*(q[1]*q[2] + q[0]*q[3])
        R31 = 2*(-q[1]*q[3] + q[0]*q[2])
        R32 = 2*(q[2]*q[3] + q[0]*q[1])
        R33 = -1*(2*q[2]**2 - 1 + 2*q[3]**2)

        phi = math.atan2(R32, R11)
        theta = math.asin(R31)
        psi = math.atan2(R21, R33)

        euler = np.array([phi, theta, psi])
        return euler

    def euler2quatern(self, euler):
        q=np.zeros(4)
        phi = euler[0]
        theta = euler[1]
        psi = euler[2]
        
        q[0] = math.cos(phi/2)*math.cos(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.sin(theta/2)*math.sin(psi/2)
        q[1] = math.sin(phi/2)*math.cos(theta/2)*math.cos(psi/2) - math.cos(phi/2)*math.sin(theta/2)*math.sin(psi/2)
        q[2] = math.cos(phi/2)*math.sin(theta/2)*math.cos(psi/2) + math.sin(phi/2)*math.cos(theta/2)*math.sin(psi/2)
        q[3] = math.cos(phi/2)*math.cos(theta/2)*math.sin(psi/2) - math.sin(phi/2)*math.sin(theta/2)*math.cos(psi/2)        
        return q
    
    def save_data(self):
        states_data = pd.DataFrame(self.drone_states)
        cmd_data = pd.DataFrame(np.hstack((self.position_cmd, self.velocity_cmd,
                                           self.attitude_cmd, self.rate_cmd,
                                           self.T, self.M)))
        states_data.columns = ['x','y','z','v_x','v_y','v_z','phi','theta','psi','w_x','w_y','w_z']
        cmd_data.columns = ['x','y','z','v_x','v_y','v_z','phi','theta','psi','w_x','w_y','w_z','T','M_x','M_y','M_z']
        current_dir = os.path.dirname(os.path.abspath(__file__))
        states_data.to_csv(os.path.join(current_dir, '../data', 'drone_state.csv'))
        cmd_data.to_csv(os.path.join(current_dir, '../data', 'drone_cmd.csv'))

if __name__ == "__main__":
    drone = DroneControlSim()
    drone.attitude_cmd[3000:, 2]=0.5   #yaw
    drone.position_cmd[3000:, 1]=10   #position step
    drone.position_cmd[5000:, 0]=10   #position step    
    drone.position_cmd[7000:, 2]=-10   #position step
    #drone.velocity_cmd[3000:, 1]=0.5   #velocity step    
    #drone.attitude_cmd[3000:]=[0.1,0.1,0.7]    #attitude step
    #drone.rate_cmd[3000:]=[0.1,0,0]    #rate step
    drone.run()
    drone.save_data()
    drone.plot_states()
    plt.show()
