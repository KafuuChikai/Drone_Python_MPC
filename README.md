# Drone_Python_MPC

This project implements **Nonlinear Model Predictive Control (NMPC)** for drones using **Python**. The goal of this project is to implement trajectory tracking and control for drones using Model Predictive Control (MPC) algorithms. The project includes implementations of various drone models.

## 1. Installation

1. Install **acados** and its dependencies, refer to the [acados official documentation](https://docs.acados.org/installation/index.html).

2. Clone this repository:

   ```
   git clone https://github.com/KafuuChikai/Drone_Python_MPC.git
   cd Drone_Python_MPC
   ```

## 2. File Structure

```plaintext
drone_model/                : Directory containing different drone models.
  ├── drone_model.py        : Full drone model.
  ├── drone_model_s.py      : Simple drone model.
  ├── drone_model_s_drag.py : Drone model with drag force.
  ├── drone_model_s_drag_delay.py : Drone model with drag force and delay.
data/                       : Directory for storing generated data.
  ├── drone_track.csv       : CSV file containing drone trajectory data.
  ├── drone_state.csv       : CSV file containing drone state data.
  ├── drone_control.csv     : CSV file containing drone control data.
acados_models/              : Directory for storing generated acados model files.
c_generated_code/           : Directory for storing generated c code.
[drone_opt_track.py]
[drone_opt_simple_point.py]
[drone_opt_simple_track.py]
[drone_opt_s_drag_track.py]
[drone_opt_s_drag_delay_track.py]
```

- `drone_opt_track.py`: Full model drone trajectory tracking.
- `drone_opt_simple_point.py`: Simple drone optimization to reach a point.
- `drone_opt_simple_track.py`: Simple drone trajectory tracking optimization.
- `drone_opt_s_drag_track.py`: Drone trajectory tracking with drag force.
- `drone_opt_s_drag_delay_track.py`: Drone trajectory tracking with drag force and delay.

## 3. Usage

1. Run the example script:
    ```bash
    python drone_opt_s_drag_delay_track.py
    ```

2. You can modify the parameters and models in the script as needed to achieve different control effects.

3. The generated data will be saved in the `data/` directory in various formats such as `.csv` files, including `drone_track.csv`, `drone_state.csv`, and `drone_control.csv`.

