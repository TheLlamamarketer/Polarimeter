import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint



def S_y(t, angle, speed):
    return -1/2 * 9.80665 * t**2 + np.sin(angle*np.pi/180) * speed * t

def S_x(t, angle, speed):
    return np.cos(angle*np.pi/180) * speed * t

def S_without_air(time, angle, speed):
    def x_t(t): return S_x(t, angle, speed)
    s_x = x_t(time)

    def y_t(t): return S_y(t, angle, speed)
    s_y = y_t(time)

    def t_x(x): return x / (np.cos(angle*np.pi/180) * speed)
    def y_x(x): return y_t(t_x(x))

    root_x = fsolve(y_x, 50000)

    # Calculate velocities directly using the derivatives
    vx_at_y_zero = np.cos(angle*np.pi/180) * speed
    vy_at_y_zero = np.sin(angle*np.pi/180) * speed
    velocity_at_y_zero = np.sqrt(vx_at_y_zero**2 + vy_at_y_zero**2)
    time_at_0 = t_x(root_x)

    return s_x, s_y, root_x, velocity_at_y_zero, time_at_0


def S_with_air(state, t, g, k, m):
    # Unpack the state vector
    x, y, vx, vy = state

    # Calculate the derivatives
    dxdt = vx
    dydt = vy
    dvxdt = -(k/m) * np.sqrt(vx**2 + vy**2) * vx
    dvydt = -g - (k/m) * np.sqrt(vx**2 + vy**2) * vy

    return [dxdt, dydt, dvxdt, dvydt]

def solve_S_with_air(angle, speed, mass, k, time_end=25, num_points=1000):
    # Initial conditions
    initial_state = [0, 0, np.cos(np.radians(angle)) * speed, np.sin(np.radians(angle)) * speed]

    # Time points
    t = np.linspace(0, time_end, num_points)

    # Solve the differential equation
    solution = odeint(S_with_air, initial_state, t, args=(9.80665, k, mass))

    # Extract the x and y coordinates
    x = solution[:, 0]
    y = solution[:, 1]

    return x, y

def find_root(angle, speed, mass, k):
    x_values, y_values = solve_S_with_air(angle, speed, mass, k)

    root_indices = np.where(np.diff(np.sign(y_values)))[0]

    for root_index in root_indices:
        x_root = np.interp(0, y_values[root_index:root_index + 2], x_values[root_index:root_index + 2])
        if x_root != 0:
            root_x = x_root
            return root_x

    return None  # No such root found

    return roots_x

def find_velocity(angle, speed, mass, k, time_end=25, num_points=1000):
    x_values, y_values = solve_S_with_air(angle, speed, mass, k, time_end, num_points)

    # Find where the trajectory crosses the x-axis (y=0)
    root_indices = np.where(np.diff(np.sign(y_values)))[0]

    if len(root_indices) > 1:
        root_index = root_indices[1]
        t_at_y_zero = np.interp(0, y_values[root_index:root_index+2], x_values[root_index:root_index+2])
        
        # Solve the differential equation again to get velocities
        solution = odeint(S_with_air, [0, 0, np.cos(np.radians(angle)) * speed, np.sin(np.radians(angle)) * speed], np.linspace(0, t_at_y_zero, num_points), args=(9.80665, k, mass))
        
        # Calculate velocity at y=0
        vx_at_y_zero = np.interp(t_at_y_zero, np.linspace(0, t_at_y_zero, num_points), solution[:, 2])
        vy_at_y_zero = np.interp(t_at_y_zero, np.linspace(0, t_at_y_zero, num_points), solution[:, 3])
        velocity_at_y_zero = np.sqrt(vx_at_y_zero**2 + vy_at_y_zero**2)

        return velocity_at_y_zero
    else:
        return None  
    
def find_time_0(angle, speed, mass, k, time_end=25, num_points=1000):
    x_values, y_values = solve_S_with_air(angle, speed, mass, k, time_end, num_points)

    # Find where the trajectory crosses the x-axis
    root_indices = np.where(np.diff(np.sign(y_values)))[0]
    root_index = root_indices[1]
    # Obtain the time values corresponding to the trajectory
    t_values = np.linspace(0, time_end, num_points)
    
    # Interpolate to find the time at which x=0
    t_at_x_zero = np.interp(0, y_values[root_index:root_index+2], t_values[root_index:root_index+2])
    
    return t_at_x_zero




angle = 45  
speed = 250   
mass = 6.0

ρ_fe = 7874
ρ_L = 1.23
Cw = 0.45
r = ((4*mass)/(3*np.pi*ρ_fe))**(1/3)
A = np.pi*r**2

k = 1/2 * A * ρ_L * Cw

x, y = solve_S_with_air(angle, speed, mass, k)


s_x, s_y, root_t, speed_at_y_zero, time_at_0 = S_without_air(np.linspace(0, 37, 300), angle, speed)




print("Without Air Resistance")
print("Root of s(x):", root_t)
print("Velocity at y=0:", speed_at_y_zero)
print("Time of impact:", time_at_0)


print("With Air Resistance")
root_x = find_root(angle, speed, mass, k)
print("Root of s(x):", root_x)

velocity_at_y_zero = find_velocity(angle, speed, mass, k)
print("Velocity at y=0:", velocity_at_y_zero)

time_0 = find_time_0(angle, speed, mass, k)
print("Time at x=0:", time_0)





def plot():

    plt.figure()

    plt.subplot(211)
    plt.plot(s_x, s_y, color='blue', label='Shot without air Resistance')
    plt.scatter(root_t, 0, color='red', label=f'Distance at which the ground is hit: {root_t[0]:.2f}m')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel('Distance travelled')
    plt.ylabel("Height")
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    plt.plot(x, y, color='green', label='Shot with air Resistance')
    plt.scatter(root_x, 0, color='red', label=f'Distance at which the ground is hit: {root_x:.2f}m')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel('Distance travelled')
    plt.ylabel("Height")
    plt.legend()
    plt.grid(True)
    
    plt.show()

plot()