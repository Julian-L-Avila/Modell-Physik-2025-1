---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Lab 1
Julian Avila

Camilo Huertas

```python
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

def F_ext(t, y):
  return 0.0

def f(t, y, k, p=2):
  return np.array([y[1], F_ext(t, y) - k * np.sign(y[0]) * np.abs(y[0])**(p - 1)])

def rk2_step(t, h, y, k, p):
  k1 = h * f(t, y, k, p)
  k2 = h * f(t + h / 2, y + k1 / 2, k, p)
  return y + (k1 + k2) / 2

def rk2_solver(a, b, n, k, p, y0):
  h = (b - a) / n
  t_values = np.linspace(a, b, n + 1)
  y_values = np.zeros((n + 1, 2))
  y_values[0] = y0

  for i in range(n):
    y_values[i + 1] = rk2_step(t_values[i], h, y_values[i], k, p)

  return t_values, y_values[:, 0], y_values[:, 1]

def plot_results(t_values, y_values, y_derivative):
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  axes[0].plot(t_values, y_values, color='gold')
  axes[0].set_title("RK2 - y(t)")
  axes[0].set_xlabel("t")
  axes[0].set_ylabel("y(t)")
  axes[0].grid(True)

  axes[1].plot(t_values, y_derivative, color='red')
  axes[1].set_title("RK2 - y'(t)")
  axes[1].set_xlabel("t")
  axes[1].set_ylabel("y'(t)")
  axes[1].grid(True)

  plt.tight_layout()
  plt.show()

def interactive_rk2(a=0, b=10, n=1000, k=1, p=2, y0_1=3, y0_2=-5):
  y0 = np.array([y0_1, y0_2])
  t_values, y_values, y_derivative = rk2_solver(a, b, n, k, p, y0)
  plot_results(t_values, y_values, y_derivative)

widgets.interactive(
  interactive_rk2,
  n=widgets.IntSlider(min=100, max=5000, step=100, value=1000, description='n'),
  k=widgets.FloatSlider(min=0.1, max=5, step=0.1, value=1, description='k'),
  p=widgets.IntSlider(min=2, max=10, step=2, value=2, description='p'),
  y0_1=widgets.FloatSlider(min=-10, max=10, step=0.1, value=3, description='y0_1'),
  y0_2=widgets.FloatSlider(min=-10, max=10, step=0.1, value=-5, description='y0_2')
)
```

# Part 2

```python
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

def f(t, y, k, m):
  return np.array([y[1], - (k / m) * y[0]])

def rk2_step(t, h, y, k, m):
  k1 = h * f(t, y, k, m)
  k2 = h * f(t + h / 2, y + k1 / 2, k, m)
  return y + (k1 + k2) / 2

def rk2_solver(a, b, n, k, m, y0):
  n = max(n, 1)
  h = (b - a) / n
  t_values = np.linspace(a, b, n + 1)
  y_values = np.empty((n + 1, 2))
  y_values[0] = y0

  for i in range(n):
    y_values[i + 1] = rk2_step(t_values[i], h, y_values[i], k, m)

  return t_values, y_values[:, 0], y_values[:, 1]

def analytical_solution(t, A, k, m):
  omega = np.sqrt(k / m)
  return A * np.cos(omega * t), -A * omega * np.sin(omega * t)

def plot_results(t_values, y_values, v_values, t_analytical, y_analytical, v_analytical):
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  axes[0].plot(t_values, y_values, label="RK2 y(t)", color='red')
  axes[0].plot(t_analytical, y_analytical, label="Analytical y(t)", linestyle="dashed", color='blue')
  axes[0].set_title("Position: RK2 vs Analytical")
  axes[0].set_xlabel("t")
  axes[0].set_ylabel("y(t)")
  axes[0].legend()
  axes[0].grid(True)

  axes[1].plot(t_values, v_values, label="RK2 v(t)", color='gold')
  axes[1].plot(t_analytical, v_analytical, label="Analytical v(t)", linestyle="dashed", color='green')
  axes[1].set_title("Velocity: RK2 vs Analytical")
  axes[1].set_xlabel("t")
  axes[1].set_ylabel("v(t)")
  axes[1].legend()
  axes[1].grid(True)

  error = np.abs((y_values - y_analytical) / y_analytical)
  axes[2].plot(t_values, error, label="Absolute Relative Error", color='purple')
  axes[2].set_title("Absolute Relative Error in Position")
  axes[2].set_xlabel("t")
  axes[2].set_ylabel("Error")
  axes[2].legend()
  axes[2].grid(True)

  plt.tight_layout()
  plt.show()

def interactive_rk2(a=0, b=10, n=1000, k=1, m=1, A=3):
  y0 = np.array([A, 0])
  t_values, y_values, v_values = rk2_solver(a, b, n, k, m, y0)
  t_analytical = np.linspace(a, b, max(n, 1) + 1)
  y_analytical, v_analytical = analytical_solution(t_analytical, A, k, m)
  plot_results(t_values, y_values, v_values, t_analytical, y_analytical, v_analytical)

widgets.interactive(
  interactive_rk2,
  n=widgets.IntSlider(min=10, max=5000, step=10, value=1000, description='n'),
  k=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=1, description='k'),
  m=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=1, description='m'),
  A=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=3, description='A')
)
```

```python
import numpy as np
import matplotlib.pyplot as plt

def f(t, y, k, m):
  return np.array([y[1], - (k / m) * y[0]])

def rk2_step(t, h, y, k, m):
  k1 = h * f(t, y, k, m)
  k2 = h * f(t + h / 2, y + k1 / 2, k, m)
  return y + (k1 + k2) / 2

def rk2_solver(a, b, n, k, m, y0):
  h = (b - a) / n
  t_values = np.linspace(a, b, n + 1)
  y_values = np.empty((n + 1, 2))
  y_values[0] = y0

  for i in range(n):
    y_values[i + 1] = rk2_step(t_values[i], h, y_values[i], k, m)

  return t_values, y_values[:, 0]

def analytical_solution(t, A, k, m):
  omega = np.sqrt(k / m)
  return A * np.cos(omega * t)

def cumulative_absolute_error(a, b, n, k, m, A):
  y0 = np.array([A, 0])
  t_values, y_values = rk2_solver(a, b, n, k, m, y0)
  y_analytical = analytical_solution(t_values, A, k, m)
  cum_error = np.cumsum(np.abs(y_values - y_analytical))
  return t_values, cum_error

def plot_cumulative_error(a=0, b=10, k=1, m=1, A=3, n_values=[20, 100, 500, 2000]):
  plt.figure(figsize=(10, 6))

  colors = ['r', 'g', 'b', 'm']
  linestyles = ['-', '--', '-.', ':']

  for i, n in enumerate(n_values):
    t_values, cum_error = cumulative_absolute_error(a, b, n, k, m, A)
    plt.plot(t_values, cum_error, label=f"n={n}", color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=2)

  plt.xlabel("Time (t)")
  plt.ylabel("Cumulative Absolute Error")
  plt.title("Cumulative Error Over Time for Different Step Sizes")
  plt.legend()
  plt.grid(True)
  plt.show()

plot_cumulative_error()
```

#Part 3

```python
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import time

def analytical_solution(t, A, k, m):
    omega = np.sqrt(k / m)
    return A * np.cos(omega * t)

def f(t, y, k, m):
    return np.array([y[1], -k / m * y[0]])

def rk2_step(t, h, y, k, m):
    k1 = h * f(t, y, k, m)
    k2 = h * f(t + h / 2, y + k1 / 2, k, m)
    return y + (k1 + k2) / 2

def rk4_step(t, h, y, k, m):
    k1 = h * f(t, y, k, m)
    k2 = h * f(t + h / 2, y + k1 / 2, k, m)
    k3 = h * f(t + h / 2, y + k2 / 2, k, m)
    k4 = h * f(t + h, y + k3, k, m)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def rk45_step(t, h, y, k, m):
    global rk45_steps
    rk45_steps += 1
    k1 = h * f(t, y, k, m)
    k2 = h * f(t + h / 4, y + k1 / 4, k, m)
    k3 = h * f(t + 3 * h / 8, y + 3 * k1 / 32 + 9 * k2 / 32, k, m)
    k4 = h * f(t + 12 * h / 13, y + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197, k, m)
    k5 = h * f(t + h, y + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104, k, m)
    k6 = h * f(t + h / 2, y - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40, k, m)
    return y + (16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55)

def solve_ode(method, a, b, n, k, m, y0):
    h = (b - a) / n
    t_values = np.linspace(a, b, n + 1)
    y_values = np.zeros((n + 1, 2))
    y_values[0] = y0
    for i in range(n):
        y_values[i + 1] = method(t_values[i], h, y_values[i], k, m)
    return t_values, y_values[:, 0], y_values[:, 1]

def interactive_comparison(A=1, k=1, m=1, n=1000):
    global rk45_steps
    rk45_steps = 0
    a, b = 0, 10
    y0 = np.array([A, 0])

    start = time.time()
    t_rk2, y_rk2, v_rk2 = solve_ode(rk2_step, a, b, n, k, m, y0)
    time_rk2 = time.time() - start

    start = time.time()
    t_rk4, y_rk4, v_rk4 = solve_ode(rk4_step, a, b, n, k, m, y0)
    time_rk4 = time.time() - start

    start = time.time()
    t_rk45, y_rk45, v_rk45 = solve_ode(rk45_step, a, b, n, k, m, y0)
    time_rk45 = time.time() - start

    t_analytical = np.linspace(a, b, n + 1)
    y_analytical = analytical_solution(t_analytical, A, k, m)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(t_rk2, y_rk2, label='RK2', linestyle='dashed')
    axes[0].plot(t_rk4, y_rk4, label='RK4', linestyle='dotted')
    axes[0].plot(t_rk45, y_rk45, label='RK45', linestyle='dashdot')
    axes[0].plot(t_analytical, y_analytical, label='Analytical', color='black')
    axes[0].set_title("Position vs Time")
    axes[0].legend()

    axes[1].plot(t_rk2, v_rk2, label='RK2', linestyle='dashed')
    axes[1].plot(t_rk4, v_rk4, label='RK4', linestyle='dotted')
    axes[1].plot(t_rk45, v_rk45, label='RK45', linestyle='dashdot')
    axes[1].set_title("Velocity vs Time")
    axes[1].legend()

    axes[2].bar(['RK2', 'RK4', 'RK45'], [time_rk2, time_rk4, time_rk45])
    axes[2].set_title("Computation Time")

    plt.show()
    print(f"RK45 Step Changes: {rk45_steps}")

widgets.interactive(
    interactive_comparison,
    A=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=1, description='A'),
    k=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=1, description='k'),
    m=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=1, description='m'),
    n=widgets.IntSlider(min=100, max=5000, step=100, value=1000, description='n')
)
```

```python
def cumulative_absolute_error(t_values, y_numerical, y_analytical):
    return np.cumsum(np.abs(y_numerical - y_analytical))

def plot_cumulative_error(method, solver, a, b, n, k, m, y0):
    t_values, y_values, _ = solver(a, b, n, k, m, y0)
    analytical = analytical_solution(t_values, y0[0], k, m)
    error = cumulative_absolute_error(t_values, y_values, analytical)

    plt.figure(figsize=(8, 5))
    plt.plot(t_values, error, label=f'{method} Cumulative Error', color='blue' if method == 'RK4' else 'red')
    plt.xlabel('Time t')
    plt.ylabel('Cumulative Absolute Error')
    plt.title(f'Cumulative Absolute Error - {method}')
    plt.legend()
    plt.grid(True)
    plt.show()

a, b = 0, 10
k, m = 1, 1
y0 = np.array([1, 0])
n = 10

plot_cumulative_error("RK4", rk4_solver, a, b, n, k, m, y0)

plot_cumulative_error("RK45", rk45_solver, a, b, n, k, m, y0)
```

#Part 4

```python
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

def eq1(t, y):
    return np.array([y[1], (y[1]**2 - y[0]**2) / (2 * y[0])])

def eq2(t, y):
    return np.array([y[1], -6 * y[0]**5])

def rk4_solver(f, a, b, n, y0):
    h = (b - a) / n
    t_values = np.linspace(a, b, n + 1)
    y_values = np.zeros((n + 1, 2))
    y_values[0] = y0

    for i in range(n):
        k1 = h * f(t_values[i], y_values[i])
        k2 = h * f(t_values[i] + h / 2, y_values[i] + k1 / 2)
        k3 = h * f(t_values[i] + h / 2, y_values[i] + k2 / 2)
        k4 = h * f(t_values[i] + h, y_values[i] + k3)
        y_values[i + 1] = y_values[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values[:, 0]

def rk45_solver(f, a, b, tol, y0):
    t = a
    y = y0
    t_values = [t]
    y_values = [y[0]]
    h = (b - a) / 100  # Paso inicial
    ops = 0

    while t < b:
        if t + h > b:
            h = b - t

        k1 = h * f(t, y)
        k2 = h * f(t + h / 4, y + k1 / 4)
        k3 = h * f(t + 3 * h / 8, y + 3 * k1 / 32 + 9 * k2 / 32)
        k4 = h * f(t + 12 * h / 13, y + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197)
        k5 = h * f(t + h, y + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104)
        k6 = h * f(t + h / 2, y - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40)

        y4 = y + (25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5)
        y5 = y + (16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55)

        error = np.linalg.norm(y5 - y4)

        if error < tol:
            t += h
            y = y5
            t_values.append(t)
            y_values.append(y[0])

        h *= min(max(0.84 * (tol / error) ** 0.25, 0.1), 4.0)
        ops += 6

    return np.array(t_values), np.array(y_values), ops

def analytical_solution(t):
    return 1 + np.sin(t)

a, b = 0, 10
y0 = np.array([1, 1])
n = 1000
tol = 1e-6

start = time.time()
t_rk4, y_rk4 = rk4_solver(eq1, a, b, n, y0)
time_rk4 = time.time() - start

start = time.time()
t_rk45, y_rk45, ops_rk45 = rk45_solver(eq1, a, b, tol, y0)
time_rk45 = time.time() - start

y_exact = analytical_solution(t_rk4)
error_rk4 = np.abs(y_rk4 - y_exact)
error_rk45 = np.abs(y_rk45 - analytical_solution(t_rk45))

data = pd.DataFrame({
    "Method": ["RK4", "RK45"],
    "Max Error": [np.max(error_rk4), np.max(error_rk45)],
    "Execution Time (s)": [time_rk4, time_rk45],
    "Operations": [n * 4, ops_rk45]
})

print(data)

plt.figure(figsize=(10, 5))
plt.plot(t_rk4, error_rk4, label="RK4 Error", color="blue")
plt.plot(t_rk45, error_rk45, label="RK45 Error", color="red")
plt.xlabel("Time t")
plt.ylabel("Absolute Error")
plt.title("Error Comparison of RK4 and RK45")
plt.legend()
plt.grid()
plt.show()
```
