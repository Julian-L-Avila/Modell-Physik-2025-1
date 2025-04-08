# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
## RK2

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

g = 9.81

def system_dynamics(t, y, m, r, alpha, beta, J):
    angle, angular_velocity = y
    torque = -m * g * r * np.cos(alpha + angle) - beta * angular_velocity
    return np.array([angular_velocity, torque / J])

def rk2_step(t, h, y, m, r, alpha, beta, J):
    k1 = h * system_dynamics(t, y, m, r, alpha, beta, J)
    k2 = h * system_dynamics(t + h/2, y + k1/2, m, r, alpha, beta, J)
    return y + k2

def rk2_solver(a, b, n, y0, m, r, alpha, beta, J):
    h = (b - a) / max(n, 1)
    t = np.linspace(a, b, n + 1)
    y = np.empty((n + 1, 2))
    y[0] = y0
    step = lambda i: rk2_step(t[i], h, y[i], m, r, alpha, beta, J)
    for i in range(n):
        y[i + 1] = step(i)
    return t, y[:, 0], y[:, 1]

def plot_solution(m, r, alpha_deg, beta, J, y0_angle_deg, y0_velocity):
    a, b, n = 0.0, 10.0, 500
    alpha = np.radians(alpha_deg)
    y0 = np.array([np.radians(y0_angle_deg), y0_velocity])
    t, y, dy = rk2_solver(a, b, n, y0, m, r, alpha, beta, J)

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(t, y, color='blue')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y(t)')
    axs[0].set_title('Angle vs Time')

    axs[1].plot(t, dy, color='green')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('y\'(t)')
    axs[1].set_title('Angular Velocity vs Time')

    axs[2].plot(y, dy, color='red')
    axs[2].set_xlabel('y(t)')
    axs[2].set_ylabel('y\'(t)')
    axs[2].set_title('Phase Space: y\'(t) vs y(t)')

    plt.tight_layout()
    plt.show()

m_slider = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Mass m')
r_slider = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Radius r')
alpha_slider = widgets.FloatSlider(value=0.0, min=-180.0, max=180.0, step=1.0, description='Alpha (deg)')
beta_slider = widgets.FloatSlider(value=0.1, min=0.0, max=5.0, step=0.1, description='Damping beta')
J_slider = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Inertia J')
y0_angle_slider = widgets.FloatSlider(value=0.0, min=0.0, max=180.0, step=1.0, description='y0 Angle (deg)')
y0_velocity_slider = widgets.FloatSlider(value=0.0, min=-10.0, max=10.0, step=0.1, description='y0 Velocity')

ui = widgets.VBox([
    m_slider, r_slider, alpha_slider, beta_slider, J_slider,
    y0_angle_slider, y0_velocity_slider
])

out = widgets.interactive_output(
    plot_solution,
    {
        'm': m_slider,
        'r': r_slider,
        'alpha_deg': alpha_slider,
        'beta': beta_slider,
        'J': J_slider,
        'y0_angle_deg': y0_angle_slider,
        'y0_velocity': y0_velocity_slider
    }
)

display(ui, out)

# %% [markdown]
## RK4

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

g = 9.81

def system_dynamics(t, y, m, r, alpha, beta, J):
    angle, angular_velocity = y
    torque = -m * g * r * np.cos(alpha + angle) - beta * angular_velocity
    return np.array([angular_velocity, torque / J])

def rk4_step(t, h, y, m, r, alpha, beta, J):
    k1 = h * system_dynamics(t, y, m, r, alpha, beta, J)
    k2 = h * system_dynamics(t + h/2, y + k1/2, m, r, alpha, beta, J)
    k3 = h * system_dynamics(t + h/2, y + k2/2, m, r, alpha, beta, J)
    k4 = h * system_dynamics(t + h, y + k3, m, r, alpha, beta, J)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def rk4_solver(a, b, n, y0, m, r, alpha, beta, J):
    h = (b - a) / max(n, 1)
    t = np.linspace(a, b, n + 1)
    y = np.empty((n + 1, 2))
    y[0] = y0
    step = lambda i: rk4_step(t[i], h, y[i], m, r, alpha, beta, J)
    for i in range(n):
        y[i + 1] = step(i)
    return t, y[:, 0], y[:, 1]

def plot_solution(m, r, alpha_deg, beta, J, y0_angle_deg, y0_velocity):
    a, b, n = 0.0, 10.0, 500
    alpha = np.radians(alpha_deg)
    y0 = np.array([np.radians(y0_angle_deg), y0_velocity])
    t, y, dy = rk4_solver(a, b, n, y0, m, r, alpha, beta, J)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(t, y, color='blue')
    axs[0].set_title('Angle θ(t)')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Angle [rad]')

    axs[1].plot(t, dy, color='green')
    axs[1].set_title('Angular Velocity θ\'(t)')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Angular Velocity [rad/s]')

    axs[2].plot(y, dy, color='red')
    axs[2].set_title('Phase Space (θ vs θ\')')
    axs[2].set_xlabel('Angle [rad]')
    axs[2].set_ylabel('Angular Velocity [rad/s]')

    plt.tight_layout()
    plt.show()

m_slider = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Mass m')
r_slider = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='r')
alpha_slider = widgets.FloatSlider(value=0.0, min=-180, max=180, step=1, description='Alpha (°)')
beta_slider = widgets.FloatSlider(value=0.5, min=0.0, max=5.0, step=0.1, description='Damping β')
J_slider = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Inertia J')

y0_angle_slider = widgets.FloatSlider(value=10.0, min=0.0, max=180, step=1, description='Initial θ (°)')
y0_velocity_slider = widgets.FloatSlider(value=0.0, min=-10.0, max=10.0, step=0.1, description='Initial θ\' (rad/s)')

ui = widgets.VBox([
    m_slider, r_slider, alpha_slider, beta_slider, J_slider,
    y0_angle_slider, y0_velocity_slider
])

out = widgets.interactive_output(
    plot_solution,
    {
        'm': m_slider,
        'r': r_slider,
        'alpha_deg': alpha_slider,
        'beta': beta_slider,
        'J': J_slider,
        'y0_angle_deg': y0_angle_slider,
        'y0_velocity': y0_velocity_slider
    }
)

display(ui, out)

# %% [markdown]
## RK45

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from scipy.integrate import solve_ivp

g = 9.81

def system_dynamics(t, y, m, r, alpha, beta, J):
    angle, angular_velocity = y
    torque = -m * g * r * np.cos(alpha + angle) - beta * angular_velocity
    return [angular_velocity, torque / J]

def rk45_solver(a, b, y0, m, r, alpha, beta, J):
    fun = lambda t, y: system_dynamics(t, y, m, r, alpha, beta, J)
    sol = solve_ivp(fun, (a, b), y0, method='RK45', t_eval=np.linspace(a, b, 500))
    return sol.t, sol.y[0], sol.y[1]

def plot_solution(m, r, alpha_deg, beta, J, y0_angle_deg, y0_velocity):
    a, b = 0.0, 10.0
    alpha = np.radians(alpha_deg)
    y0 = [np.radians(y0_angle_deg), y0_velocity]

    t, y, dy = rk45_solver(a, b, y0, m, r, alpha, beta, J)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(t, y, color='blue')
    axs[0].set_title('Angle vs Time')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Angle (rad)')

    axs[1].plot(t, dy, color='green')
    axs[1].set_title('Angular Velocity vs Time')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Angular Velocity (rad/s)')

    axs[2].plot(y, dy, color='red')
    axs[2].set_title('Phase Space (Angle vs Angular Velocity)')
    axs[2].set_xlabel('Angle (rad)')
    axs[2].set_ylabel('Angular Velocity (rad/s)')

    plt.tight_layout()
    plt.show()

m_widget = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Mass (kg)')
r_widget = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Radius (m)')
alpha_widget = widgets.FloatSlider(value=0.0, min=-180.0, max=180.0, step=1.0, description='Alpha (deg)')
beta_widget = widgets.FloatSlider(value=0.1, min=0.0, max=5.0, step=0.1, description='Damping (kg m²/s)')
J_widget = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Inertia (kg m²)')

y0_angle_widget = widgets.FloatSlider(value=10.0, min=0.0, max=180.0, step=1.0, description='y₀ (deg)')
y0_velocity_widget = widgets.FloatSlider(value=0.0, min=-10.0, max=10.0, step=0.1, description="y'₀ (rad/s)")

ui = widgets.VBox([
    m_widget,
    r_widget,
    alpha_widget,
    beta_widget,
    J_widget,
    y0_angle_widget,
    y0_velocity_widget
])

out = widgets.interactive_output(
    plot_solution,
    {
        'm': m_widget,
        'r': r_widget,
        'alpha_deg': alpha_widget,
        'beta': beta_widget,
        'J': J_widget,
        'y0_angle_deg': y0_angle_widget,
        'y0_velocity': y0_velocity_widget
    }
)

display(ui, out)

# %% [markdown]
## Spectrum

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

g = 9.81

def system_dynamics(t, y, m, r, alpha, beta, J):
    angle, angular_velocity = y
    torque = -m * g * r * np.cos(alpha + angle) - beta * angular_velocity
    return angular_velocity, torque / J

def solve_motion(a, b, y0, m, r, alpha, beta, J):
    dynamics = lambda t, y: system_dynamics(t, y, m, r, alpha, beta, J)
    t_span = np.linspace(a, b, 500)
    sol = solve_ivp(dynamics, (a, b), y0, method='RK45', t_eval=t_span)
    return sol.t, sol.y[0], sol.y[1]

def compute_spectrum(signal, sampling_rate):
    n = signal.size
    frequencies = fftfreq(n, d=1/sampling_rate)
    spectrum = np.abs(fft(signal)) / n
    return frequencies[:n // 2], spectrum[:n // 2]

def plot_all(m, r, alpha_deg, beta, J, y0_angle_deg, y0_velocity):
    a, b = 0.0, 10.0
    alpha = np.radians(alpha_deg)
    y0 = (np.radians(y0_angle_deg), y0_velocity)
    t, y, dy = solve_motion(a, b, y0, m, r, alpha, beta, J)
    frequencies, spectrum = compute_spectrum(y, sampling_rate=t.size / (b - a))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(t, y, color='blue')
    axs[0, 0].set_title('Angle vs Time')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Angle (rad)')

    axs[0, 1].plot(t, dy, color='green')
    axs[0, 1].set_title('Angular Velocity vs Time')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Angular Velocity (rad/s)')

    axs[1, 0].plot(y, dy, color='red')
    axs[1, 0].set_title('Phase Space (Angle vs Angular Velocity)')
    axs[1, 0].set_xlabel('Angle (rad)')
    axs[1, 0].set_ylabel('Angular Velocity (rad/s)')

    axs[1, 1].plot(frequencies, spectrum, color='purple')
    axs[1, 1].set_title('Fourier Spectrum of Angle')
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_ylabel('Amplitude')
    axs[1, 1].set_xlim(0, 10)

    fig.tight_layout()
    plt.show()

def create_widgets():
    m = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Mass (kg)')
    r = widgets.FloatSlider(value=0.5, min=0.1, max=5.0, step=0.1, description='Radius (m)')
    alpha_deg = widgets.FloatSlider(value=0.0, min=-180, max=180, step=1, description='Alpha (deg)')
    beta = widgets.FloatSlider(value=0.1, min=0.0, max=5.0, step=0.1, description='Damping')
    J = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Inertia')
    y0_angle_deg = widgets.FloatSlider(value=10.0, min=-180, max=180, step=1, description='Initial Angle (deg)')
    y0_velocity = widgets.FloatSlider(value=0.0, min=-10.0, max=10.0, step=0.1, description='Initial Velocity')

    ui = widgets.VBox([m, r, alpha_deg, beta, J, y0_angle_deg, y0_velocity])
    out = widgets.interactive_output(
        plot_all,
        {'m': m, 'r': r, 'alpha_deg': alpha_deg, 'beta': beta, 'J': J, 'y0_angle_deg': y0_angle_deg, 'y0_velocity': y0_velocity}
    )
    display(ui, out)

create_widgets()

