import deepxde as dde
import numpy as np
import os
import matplotlib.pyplot as plt

GM = 1.0
t_begin = 0.0
t_end = 10.0
geom = dde.geometry.TimeDomain(t_begin, t_end)

x0_val = 1.0
y0_val = 0.0
vx0_val = 0.0
vy0_val = 1.0

def boundary_initial(t_spatial, on_initial):
    return on_initial and dde.utils.isclose(t_spatial[0], t_begin)

def create_kepler_net(
    input_dims=1,
    output_dims=4,
    num_hidden_layers=3,
    num_neurons_per_layer=50,
    hidden_activation='tanh',
    kernel_initializer='Glorot uniform'
):
    layer_sizes = [input_dims]
    if isinstance(num_neurons_per_layer, int):
        if num_hidden_layers > 0:
            layer_sizes.extend([num_neurons_per_layer] * num_hidden_layers)
    elif isinstance(num_neurons_per_layer, (list, tuple)):
        if len(num_neurons_per_layer) != num_hidden_layers:
            raise ValueError(
                "Length of num_neurons_per_layer list/tuple must match num_hidden_layers"
            )
        layer_sizes.extend(num_neurons_per_layer)
    elif num_hidden_layers > 0:
        raise TypeError(
            "num_neurons_per_layer must be an int or a list/tuple of ints if num_hidden_layers > 0"
        )
    layer_sizes.append(output_dims)
    net = dde.nn.FNN(layer_sizes, hidden_activation, kernel_initializer)
    return net

def plot_loss_history(losshistory, model_name, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    epochs = np.array(losshistory.steps)
    loss_train_all_comps = np.array(losshistory.loss_train)
    total_loss_train = loss_train_all_comps.sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_loss_train, label='Total Train Loss')

    if hasattr(losshistory, 'loss_test') and losshistory.loss_test:
        loss_test_values = np.array(losshistory.loss_test)
        if loss_test_values.ndim > 0 and loss_test_values.shape[0] == len(epochs):
            # Handle if loss_test is multi-component or already total
            if loss_test_values.ndim == 2 :
                 current_loss_test = loss_test_values.sum(axis=1)
            else:
                 current_loss_test = loss_test_values.flatten()
            plt.plot(epochs, current_loss_test, label='Total Test Loss', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title(f'{model_name} - Total Loss vs. Epoch')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_total_loss_history.png"))
    plt.close()

    if loss_train_all_comps.shape[1] > 1: # If more than one loss component
        # Try to get component names if available from LossHistory, else generate defaults
        if hasattr(losshistory, 'loss_train_names') and losshistory.loss_train_names and len(losshistory.loss_train_names) == loss_train_all_comps.shape[1]:
            component_names = losshistory.loss_train_names
        else:
            component_names = [f'Train Loss Comp {i+1}' for i in range(loss_train_all_comps.shape[1])]

        for i in range(loss_train_all_comps.shape[1]):
            plt.plot(epochs, loss_train_all_comps[:, i], label=component_names[i])

        plt.xlabel('Epoch')
        plt.ylabel('Individual Loss Value')
        plt.title(f'{model_name} - Individual Training Loss Components vs. Epoch')
        plt.legend(loc='best', fontsize='small')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{model_name}_individual_loss_history.png"))
        plt.close()

if __name__ == "__main__":
    print("common_components.py recreated successfully with generalized functions.")
    net_test = create_kepler_net()
    print(f"Test net type: {type(net_test)}")
    # Dummy loss history for plotting test
    class DummyLossHistory:
        def __init__(self, steps, loss_train, loss_test, loss_train_names=None):
            self.steps = steps
            self.loss_train = loss_train
            self.loss_test = loss_test
            self.loss_train_names = loss_train_names # Optional component names

    steps = list(range(0, 1000, 100))
    # Example with 2 components for loss_train
    loss_tr = [[np.exp(-s/1000)+0.1, 0.5*np.exp(-s/1000)+0.05] for s in steps]
    loss_tr_names = ["PDE_residual_1", "BC_loss_1"] # Dummy names for components
    # Example for loss_test (usually total, so one value per epoch)
    loss_te = [[np.exp(-s/900)+0.12] for s in steps] # List of lists, one value per step

    lh_dummy = DummyLossHistory(steps, loss_tr, loss_te, loss_tr_names)
    plot_loss_history(lh_dummy, "DummyTestCommon", output_dir="common_components_test_output")
    print("Dummy loss plot for common_components.py saved in common_components_test_output/")
