#Fuel Cell PINN Model V1.2 - DeepXDE with PyTorch
#Bailey Steger - 9/16/2026
#Residual-based adaptive refinement
##########################################


import os
os.environ["DDE_BACKEND"] = "pytorch"


import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#GPU check
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())



#1) DEFINE MODEL INPUTS
rho = 998 #kg/m^3 #CONSTANT FOR NOW
mu = 0.001002 #N*s/m^2 #CONSTANT FOR NOW
u0 = 0.1 #m/s
L=0.1 #m, tube length
R=0.0075 #m, tube radius
L_R = L/R
Re = rho*u0*2*R/mu

#2) FORMAT GEOMETRY

flowbox = dde.geometry.Rectangle([0,0],[L_R,1])
# timedomain = dde.geometry.TimeDomain(0, 10)

# geom = dde.geometry.GeometryXTime(flowbox, timedomain)
geom = flowbox #allows for addition of other geometry modifications

#3) Compute Physics Equations

def physLoss(x,y):
    """
    x: Network inputs [z,r] (Coordinates)
    y: Network outputs [u_z, u_r, p] (Velocities and Pressure)
    """

    #---grab inputs and outputs---
    r = x[:,1:2] #pull radial position

    u_z = y[:,0:1] #pull z velocity component
    u_r = y[:,1:2] #pull r velocity component
    p = y[:,2:3] #pull pressure 

    #---Compute First derivatives---
    du_z_z = dde.grad.jacobian(y, x, i=0, j=0)
    du_z_r = dde.grad.jacobian(y, x, i=0, j=1)
    # du_z_t = dde.grad.jacobian(y, x, i=0, j=2)

    du_r_z = dde.grad.jacobian(y, x, i=1, j=0)
    du_r_r = dde.grad.jacobian(y, x, i=1, j=1)
    # du_r_t = dde.grad.jacobian(y, x, i=1, j=2)

    dp_z = dde.grad.jacobian(y, x, i=2, j=0)
    dp_r = dde.grad.jacobian(y, x, i=2, j=1)
    #dp_t = dde.grad.jacobian(y, x, i=2, j=2)


    #---Compute Second derivatives---
    du_z_zz = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_z_rr = dde.grad.hessian(y, x, component=0, i=1, j=1)
    du_r_zz = dde.grad.hessian(y, x, component=1, i=0, j=0)
    du_r_rr = dde.grad.hessian(y, x, component=1, i=1, j=1)

    #---Compute Losses--- (Navier Stokes, Non-Dimensionalized)
    #No time transients
    #variables in following equations are assumed to be dimensionless, so additional notation is not applied 
  

    nu_star = 2 / Re  # Radius-scaled coordinates, diameter-based Re

    lossZmom = (r * (u_r * du_z_r + u_z * du_z_z + dp_z) - nu_star * (r * (du_z_rr + du_z_zz) + du_z_r))

    lossRmom = (r**2 * (u_r * du_r_r + u_z * du_r_z + dp_r) - nu_star * (r**2 * (du_r_rr + du_r_zz)+ r * du_r_r - u_r))

    lossCont = r * (du_r_r + du_z_z) + u_r

    return [lossZmom, lossRmom, lossCont]

#4) Boundary Conditions

#Define Boundary Condition Locations

def top_wall(x, on_boundary):
    # Adjust 'y_max_coordinate' to match your actual geometry height
    return on_boundary & np.isclose(x[1], 1)

def bottom_wall(x, on_boundary):
    # Adjust 'y_min_coordinate' to match your actual geometry base
    return on_boundary & np.isclose(x[1], 0)

def left_wall(x, on_boundary):
    # Adjust 'x_min_coordinate' to match your actual geometry left edge
    return on_boundary & np.isclose(x[0], 0)

def right_wall(x, on_boundary):
    # Adjust 'x_min_coordinate' to match your actual geometry right edge
    return on_boundary & np.isclose(x[0], L_R)


#Associate boundary conditions with the walls

# --- UPPER WALL: uz = 0, ur = 0 ---
bc_top_uz = dde.icbc.DirichletBC(geom, lambda x: 0, top_wall, component=0)
bc_top_ur = dde.icbc.DirichletBC(geom, lambda x: 0, top_wall, component=1)

# --- LOWER WALL (AXIS): uz = 0, ur = 0 ---
#bc_bot_u = dde.icbc.DirichletBC(geom, lambda x: 0, bottom_wall, component=0)
bc_axis_ur = dde.icbc.DirichletBC(geom, lambda x: 0, bottom_wall, component=1)
bc_axis_uz = dde.icbc.OperatorBC(geom, lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1), bottom_wall,)
bc_axis_p = dde.icbc.OperatorBC(geom, lambda x, y, _: dde.grad.jacobian(y, x, i=2, j=1), bottom_wall,)

# --- LEFT WALL (Inlet): uz = u0, ur = 0 ---
# Replace 'u0_value' with your specific numeric flow speed (e.g., 1.0) 
bc_left_uz = dde.icbc.DirichletBC(geom, lambda x: u0, left_wall, component=0)
bc_left_ur = dde.icbc.DirichletBC(geom, lambda x: 0, left_wall, component=1)

# --- RIGHT WALL (Outlet): p=0
# Replace 'u0_value' with your specific numeric flow speed (e.g., 1.0) 
bc_right_p = dde.icbc.DirichletBC(geom, lambda x: 0, right_wall, component=2)

#Combine boundary conditions
bcs = [bc_top_uz, bc_top_ur, bc_axis_ur, bc_axis_uz, bc_axis_p, bc_left_uz, bc_left_ur, bc_right_p] 


# 5). Combine everything into a DeepXDE Data Object
data = dde.data.PDE(
    geom,
    physLoss,
    bcs,
    num_domain=5000,  # Number of collocation points inside the domain
    num_boundary=1000, # Number of points on the boundary
)

# 6). Construct the Neural Network Architecture
layer_size = [2] + [30] * 6 + [3]  # Input (x, t) -> 5 hidden layers of 20 neurons -> Output (u_z, u_r, p)
activation = "tanh"
initializer = "Glorot normal"

dde.optimizers.config.set_LBFGS_options(
    maxcor=3000,
    ftol=0,
    gtol=1e-08,
    maxiter=5000,
    maxfun=None,
    maxls=50
)

net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)


# 7). Start Training the Model W/ Adam


#Increase weights to focus on dropping specific residuals
weights = [10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1] #first 3 physics, last 8 BCs

print("Training with Adam optimizer...")
model.compile("adam", lr=0.001, loss_weights=weights)

#---Plot current collocation grid---
x_train = data.train_x 

x_coords = x_train[:, 0]
y_coords = x_train[:, 1]

plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, s=15, c='blue', alpha=0.7, label='Collocation Points')
plt.title("DeepXDE Collocation Grid")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show(block=False)
plt.pause(0.001)
#------------------------------

model.train(iterations=1000) #Adam iterations before RAR

# 8). Residual-based Adaptive Refinement W/ Adam

max_rar_iterations = 10
points_to_add_per_iteration = 400
residual_threshold = 1e-6

for i in range(max_rar_iterations):
    #---Plot collocation grid---
    x_train = data.train_x 

    x_coords = x_train[:, 0]
    y_coords = x_train[:, 1]

    plt.figure(figsize=(6, 1))
    plt.scatter(x_coords, y_coords, s=15, c='blue', alpha=0.7, label='Collocation Points')
    plt.title("DeepXDE Collocation Grid")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)
    #------------------------------
    
    # 1. Sample a large pool of random candidate points across the domain
    X_candidates = geom.random_points(4000) #define number points to sample
    
    # 2. FIX: Use model.predict with the operator argument pointing to physLoss
    # This evaluates your custom PDE equations directly
    f_residuals_list = model.predict(X_candidates, operator=physLoss)
    
    # 3. Stack the list of arrays horizontally into a single 2D array
    f_residuals = np.hstack(f_residuals_list) # Shape becomes (10000, 3)
    abs_residuals = np.abs(f_residuals)
    
    # 4. Average across your 3 physical equations (axis=1) to get 1 residual error per point
    abs_residuals = np.mean(abs_residuals, axis=1) # Shape becomes (10000,)
    mean_residual = np.mean(abs_residuals)

    print(f"RAR Iteration {i+1}: Mean Residual = {mean_residual:.5f}")
    
    # 5. Check for early termination if overall accuracy is sufficient
    if mean_residual < residual_threshold:
        print("Target residual threshold achieved. Stopping refinement.")
        break
        
    # 6. Identify the indices of points with the largest residuals (Greedy approach)
    worst_indices = np.argpartition(abs_residuals, -points_to_add_per_iteration)[-points_to_add_per_iteration:]
    new_points = X_candidates[worst_indices]
    
    # 7. Inject the worst performing points directly into the active training pool
    model.data.add_anchors(new_points)
    
    # 8. Retrain the network with the newly added constraints
    model.train(iterations=1000) #define how many training iterations between RAR sessions 




#9) L-BFGS optimizer to finely tune the physics losses
print("Fine-tuning with L-BFGS optimizer...")
model.compile("L-BFGS")
model.train()


# Save trained model to models/ folder
model.save("models/pinn_model")

# Prints exact file name for saved model
print("Saved model:")
print(max(
    (os.path.join("models", f) for f in os.listdir("models")),
    key=os.path.getmtime
))



#10) Predict and Visualize Results

# Generate a grid of points to evaluate the trained model
x_vals = np.linspace(0, L_R, 200)
y_vals = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x_vals, y_vals)
points = np.vstack((np.ravel(X), np.ravel(Y))).T

# Make predictions (returns an array of [u, v, p])
predictions = model.predict(points)

# Check predictions before plotting
for i, name in enumerate(["u_z", "u_r", "p"]):
    values = predictions[:, i]
    finite = np.isfinite(values)
    print(f"{name}: {finite.sum()}/{values.size} finite")
    if finite.any():
        print("  range:", values[finite].min(), values[finite].max())

if not np.isfinite(predictions).all():
    raise ValueError(
        "Predictions contain NaN or infinity. Check the training losses."
    )

#---Extract velocity components and reshape to grid---
U_velocity = predictions[:, 0].reshape(200, 200)
V_velocity = predictions[:, 1].reshape(200, 200)
pressure = predictions[:, 2].reshape(200, 200)*rho*u0**2
mag_velocity = (U_velocity**2 + V_velocity**2)**0.5

#---Plotting the magnitude contour---
plt.figure(figsize=(10, 2))
contour = plt.contourf(X, Y, mag_velocity, levels=100, cmap='jet')
plt.colorbar(contour, label='Velocity $u$ (m/s)')
plt.title('PINN Prediction: 2D Velocity Magnitude Contour In Pipe')
plt.xlabel('$x$ (Length)')
plt.ylabel('$y$ (Height)')
plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

#---Plotting the u_velocity---
plt.figure(figsize=(10, 2))
contour = plt.contourf(X, Y, U_velocity, levels=100, cmap='jet')
plt.colorbar(contour, label='Velocity $u$ (m/s)')
plt.title('PINN Prediction: U_Velocity Contour In Pipe')
plt.xlabel('$x$ (Length)')
plt.ylabel('$y$ (Height)')
plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

#---Plotting the v_velocity---
plt.figure(figsize=(10, 2))
contour = plt.contourf(X, Y, V_velocity, levels=100, cmap='jet')
plt.colorbar(contour, label='Velocity $u$ (m/s)')
plt.title('PINN Prediction: V_Velocity Contour In Pipe')
plt.xlabel('$x$ (Length)')
plt.ylabel('$y$ (Height)')
plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

#---Plotting the pressure---
plt.figure(figsize=(10, 2))
contour = plt.contourf(X, Y, pressure, levels=100, cmap='jet')
plt.colorbar(contour, label='Pressure $p$ (Pa)')
plt.title('PINN Prediction: Pressure Contour In Pipe')
plt.xlabel('$x$ (Length)')
plt.ylabel('$y$ (Height)')
plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)



#---Generate total loss heat map---
# Grid: inputs are [z/R, r/R]
z_vals = np.linspace(0, 5, 200)
r_vals = np.linspace(0, 1, 100)
Z, RR = np.meshgrid(z_vals, r_vals)
points = np.column_stack((Z.ravel(), RR.ravel()))

# Evaluate the three PDE residuals using automatic differentiation
residuals = model.predict(points, operator=physLoss)
residuals = np.hstack(residuals)  # Shape: (number of points, 3)

if not np.isfinite(residuals).all():
    raise ValueError(
        "PDE residuals contain NaN or infinity. Check the axis treatment."
    )

# Pointwise sum of squared PDE residuals
local_loss = np.sum(residuals.astype(np.float64)**2, axis=1)
local_loss = local_loss.reshape(Z.shape)

fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)

heatmap = ax.pcolormesh(
    R * Z,
    R * RR,
    np.maximum(local_loss, 1e-20),
    shading="auto",
    cmap="inferno",
    norm=LogNorm(),)

fig.colorbar(heatmap, ax=ax, label="Sum of squared PDE residuals")
ax.set_xlabel("Axial position z (m)")
ax.set_ylabel("Radius r (m)")
ax.set_title("Local PDE residual")
plt.show()
