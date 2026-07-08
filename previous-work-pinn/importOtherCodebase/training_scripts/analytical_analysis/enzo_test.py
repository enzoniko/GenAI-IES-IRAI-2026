from inspect import Parameter

# %%
"""
=============================================================================
CELL 2: CORE SIMULATION SETUP
=============================================================================
This cell defines all the core classes and functions required for the simulation.
It includes the equations of motion, force models, the simulation runner,
and plotting logic. Do not modify this cell unless you are changing the
fundamental physics or simulation logic.
"""
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import time

# Setup for the 1-minute timeout feature (not available on Windows)
if sys.platform != "win32":
    import signal

    class TimeoutException(Exception):
        """Custom exception for timeouts."""
        pass

    def timeout_handler(signum, frame):
        """Handler to raise the timeout exception."""
        raise TimeoutException("Simulation timed out after 60 seconds.")

    # Associate the SIGALRM signal with the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)


def nonlinear_bearing_force(X, Y, X_dot, Y_dot, params):
    """
    Calculates nonlinear bearing forces based on the short-bearing theory.
    This model captures how forces change with the journal's position (epsilon)
    and velocity (epsilon_dot) within the bearing clearance.
    """
    c = params.C_bearing
    mu = params.mu_oil
    L = params.L_bearing
    R = params.R_bearing
    Omega = params.Omega

    # Avoid division by zero if the journal is perfectly centered
    if abs(X) < 1e-12 and abs(Y) < 1e-12:
        return 0.0, 0.0

    # Calculate eccentricity and its rate of change
    epsilon = np.sqrt(X**2 + Y**2) / c
    if epsilon >= 1.0:
        epsilon = 0.999 # Prevent singularity at the boundary

    epsilon_dot = (X * X_dot + Y * Y_dot) / (c**2 * (epsilon + 1e-12))

    # Calculate radial and tangential force components
    common_factor = (mu * L**3 * R) / (2 * c**2 * (1 - epsilon**2)**2)
    Fr = common_factor * (4 * epsilon_dot)
    Ft = common_factor * Omega * epsilon * np.pi * (1 + 2 * epsilon**2)

    # Resolve forces into X and Y components
    Fx = (Fr * X - Ft * Y) / (c * epsilon)
    Fy = (Fr * Y + Ft * X) / (c * epsilon)

    return -Fx, -Fy

def equations_of_motion(t, S, params, flags):
    """
    Defines the system of first-order ordinary differential equations (ODEs).
    This function is the core of the simulation, calculating the derivatives
    of the state vector at a given time `t`.
    """
    # Unpack the state vector S into human-readable variables
    X1, Y1, X2, Y2, X3, Y3, X1_dot, Y1_dot, X2_dot, Y2_dot, X3_dot, Y3_dot = S
    Omega = params.Omega
    g = params.g
    f = flags # Use a shorter alias for the flags dictionary

    # --- Initialize forces with base linear components (shaft stiffness and damping) ---
    force_X1 = -params.D1*X1_dot - params.K1*(X1-X2) - params.K2*(X1-X3)
    force_Y1 = -params.D1*Y1_dot - params.K1*(Y1-Y2) - params.K2*(Y1-Y3)
    force_X2 = -params.D2*X2_dot - params.K1*(X2-X1)
    force_Y2 = -params.D2*Y2_dot - params.K1*(Y2-Y1)
    force_X3 = -params.D3*X3_dot - params.K2*(X3-X1)
    force_Y3 = -params.D3*Y3_dot - params.K2*(Y3-Y1)

    # --- Conditionally add forces based on the `flags` for the current test ---
    if f['imbalance']:
        force_X1 += params.M1 * params.E1 * Omega**2 * np.cos(Omega * t)
        force_Y1 += params.M1 * params.E1 * Omega**2 * np.sin(Omega * t)

    if f['gravity']:
        force_Y1 -= params.M1 * g
        force_Y2 -= params.M2 * g
        force_Y3 -= params.M3 * g

    if f['bearing_type'] == 'linear':
        force_X2 -= params.K_b * X2
        force_Y2 -= params.K_b * Y2
        force_X3 -= params.K_b * X3
        force_Y3 -= params.K_b * Y3
    elif f['bearing_type'] == 'nonlinear':
        Fx2, Fy2 = nonlinear_bearing_force(X2, Y2, X2_dot, Y2_dot, params)
        Fx3, Fy3 = nonlinear_bearing_force(X3, Y3, X3_dot, Y3_dot, params)
        force_X2 += Fx2
        force_Y2 += Fy2
        force_X3 += Fx3
        force_Y3 += Fy3

    # --- Conditionally add seal forces ---
    mass_eff = params.M1 # Effective mass of the impeller
    if f['seal_direct'] or f['seal_crosscoupled']:
        epsilon_seal = np.sqrt(X1**2 + Y1**2) / params.C_seal
        if epsilon_seal >= 1.0:
            epsilon_seal = 0.999 # Prevent singularity

        # Calculate eccentricity-dependent seal coefficients
        K_seal = params.K0 * (1 - epsilon_seal**2)**params.n1
        D_seal = params.D0 * (1 - epsilon_seal**2)**params.n2
        tau_seal = params.tau0 * (1 - epsilon_seal)**params.b

        if f['seal_direct']:
            force_X1 += -(K_seal - params.Mf * Omega**2 * tau_seal**2) * X1 - D_seal * X1_dot
            force_Y1 += -(K_seal - params.Mf * Omega**2 * tau_seal**2) * Y1 - D_seal * Y1_dot

        if f['seal_crosscoupled']:
            force_X1 += -D_seal * Omega * tau_seal * Y1 - (2 * params.Mf * Omega * tau_seal) * Y1_dot
            force_Y1 += D_seal * Omega * tau_seal * X1 + (2 * params.Mf * Omega * tau_seal) * X1_dot

        # The seal's fluid inertia adds to the impeller's mass
        mass_eff = params.M1 + params.Mf

    # --- Return the derivatives (accelerations) ---
    # This is the dS/dt vector that the ODE solver needs
    X1_ddot = force_X1 - params.D1*X1_dot - params.K2*(X1-X2) - params.K2*(X1-X2)
    Y1_ddot = force_Y1 - params.D1*Y1_dot - params.K2*(Y1-Y2) - params.K2*(Y1-Y2)
    X2_ddot = force_X2 - params.D2*X2_dot - params.K1*(X2-X1)
    Y2_ddot = force_Y2 - params.D2*Y2_dot - params.K1*(Y2-Y1)
    X3_ddot = force_X3 - params.D3*X3_dot - params.K2*(X3-X1)
    Y3_ddot = force_Y3 - params.D3*Y3_dot - params.K2*(Y3-Y1)

    X1_ddot = X1_ddot/params.M1
    Y1_ddot = Y1_ddot/params.M1
    X2_ddot = X2_ddot/params.M2
    Y2_ddot = Y2_ddot/params.M2
    X3_ddot = X3_ddot/params.M3
    Y3_ddot = Y3_ddot/params.M3


    return [
        X1_dot, Y1_dot, X2_dot, Y2_dot, X3_dot, Y3_dot,
        X1_ddot, Y1_ddot,
        X2_ddot, Y2_ddot,
        X3_ddot, Y3_ddot
    ]

import numpy as np



def run_simulation(params, flags, step_title):
    """
    A wrapper function to run the ODE solver and plot results.
    Includes the 1-minute timeout logic.
    """
    S0 = np.zeros(12)  # Initial conditions (all displacements and velocities are zero)
    t_span = [0.0, 120] # Simulation time from 0 to 0.5 seconds
    t_eval = np.linspace(t_span[0], t_span[1], 5000) # Points in time to store the solution
    solution = None

    print(f"--- Running: {step_title} ---")
    start_time = time.time()

    try:
        # Set a 60-second alarm (only on non-Windows systems)
        if sys.platform != "win32":
            signal.alarm(180)
        else:
            # Print a one-time warning on Windows
            if 'first_win_warning' not in globals():
                print("NOTE: Timeout feature is disabled on Windows.")
                globals()['first_win_warning'] = True

        # Call the ODE solver. 'Radau' is a good choice for stiff systems like this.
        solution = solve_ivp(
            fun=equations_of_motion,
            t_span=t_span,
            y0=S0,
            args=(params, flags),
            method='RK45',
            t_eval=t_eval
        )

    except TimeoutException as e:
        print(f"Simulation FAILED: {e}\n")
        return # Exit the function if it times out
    finally:
        # Always disable the alarm when the block is exited
        if sys.platform != "win32":
            signal.alarm(0)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration:.2f} seconds.")

    if solution and solution.success:
        print("Simulation successful.\n")
        # Create plots to visualize the results
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(step_title, fontsize=16)

        # Plot 1: Impeller Orbit (last 1000 points for steady state)
        ax=axes[0,0]; ax.plot(solution.y[0,-1000:]*1e3, solution.y[1,-1000:]*1e3); ax.set_title('Impeller Orbit (Steady State)'); ax.set_xlabel('X1 (mm)'); ax.set_ylabel('Y1 (mm)'); ax.grid(True); ax.axis('equal')

        # Plot 2: Bearing 1 Orbit (last 1000 points for steady state)
        ax=axes[0,1]; ax.plot(solution.y[2,-1000:]*1e3, solution.y[3,-1000:]*1e3); ax.set_title('Bearing 1 Orbit (Steady State)'); ax.set_xlabel('X2 (mm)'); ax.set_ylabel('Y2 (mm)'); ax.grid(True); ax.axis('equal')

        # Plot 3: Impeller Y-displacement history
        ax=axes[1,0]; ax.plot(solution.t, solution.y[1,:]*1e3); ax.set_title('Impeller Y1 History'); ax.set_xlabel('Time (s)'); ax.set_ylabel('Y1 (mm)'); ax.grid(True)

        # Plot 4: Bearing 1 Y-displacement history
        ax=axes[1,1]; ax.plot(solution.t, solution.y[3,:]*1e3); ax.set_title('Bearing 1 Y2 History'); ax.set_xlabel('Time (s)'); ax.set_ylabel('Y2 (mm)'); ax.grid(True)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()
    elif solution:
        print(f"Simulation FAILED: {solution.message}\n")


# %%
"""
=============================================================================
TEST CELL for Original Script: Full Model with Paper Parameters
=============================================================================
This cell is for use with the original script
('rotor_dynamics_simulation_with_progressive_diagnostics.py').
It uses the parameters from the Wang & Wang (2010) paper to test if
that script's logic converges with a known physical configuration.
"""

# Redefine the parameter function with values from the paper
def get_paper_parameters():
    class Parameters:
        def __init__(self):
            # --- Main System Properties from Wang & Wang (2010) ---
            self.M1 = 50.0
            self.M2 = 3.5
            self.M3 = 3.5
            self.K1 = 3.4635e6
            self.K2 = 3.8127e6
            self.D1 = 3000.0
            self.D2 = 3000.0
            self.D3 = 3000.0
            self.Omega = 3000.0
            self.E1 = 5.0e-6 # Using the value from the original script's Step 5.3
            self.g = 9.81

            # --- Bearing Properties ---
            self.L_bearing = 0.06
            self.R_bearing = 0.035
            self.C_bearing = 0.3e-3
            self.mu_oil = 1.47e-5 # Air viscosity from paper

            # --- Seal Properties ---
            self.C_seal = 0.3e-3
            self.Mf = 0.5 # Assumed from your script, not specified in paper
            self.K0 = 1.0e5
            self.D0 = 1.5e3
            self.tau0 = 0.4
            self.n1 = -3.0
            self.n2 = -2.0
            self.b = 0.45
    return Parameters()

if __name__ == "__main__":

    # --- Run the simulation for the full model ---
    params = get_paper_parameters()
    flags = {
        'imbalance': True,
        'gravity': True,
        'bearing_type': 'nonlinear',
        'seal_direct': True,
        'seal_crosscoupled': True # All forces are ON
    }
    # This will call the original run_simulation and dimensionless_equations_of_motion
    run_simulation(params, flags, "Step 5.3 (Original Script) with Paper Parameters")