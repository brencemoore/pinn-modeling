#Fuel Cell PINN Model V3 - DeepXDE with PyTorch
#10/22/2026
#Bailey Steger
#Brence Moore

#Species transport and energy balance
#H2 -> H20 at walls

##########################################


import os
os.environ["DDE_BACKEND"] = "pytorch"


import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from datetime import datetime
from pinn_utils import TrainingLog, model_directory
from plot_pinn_model import plot_after_training

dde.config.set_default_float("float64")


#GPU check
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())

SEED = 42
dde.config.set_random_seed(SEED)

#1) DEFINE MODEL INPUTS

#general fluid properties
rho_1 = 0.0306     #kg/m^3 #CONSTANT FOR NOW, hydrogen
rho_2 = 998      #kg/m^3, Water
mu = 0.001002    #N*s/m^2 #CONSTANT FOR NOW
uAVG = 1         #m/s, inlet velocity
T_w = 1000      #K, constant wall temperature
T_0 = 800       #K, constant inlet temperature profile
M_A = 2.016      #Molar mass H2, g/mol
M_B = 18.015     #Molar mass H20, g/mol
M_0 = M_A        #Reference molar mass, g/mol
F = 96485        #Faradays constant, coulombs/mole
R_gas = 8.314    #Universal gas constant, J/(mol*K)
p = 1.01e5       #Pa, assume constant
p_0 = p
p_e = p-1e-5
p_atm = p/1.01e5 #atm
k = 0.18 #W/(m*K), thermal conductivity (H2)
C_p1 = 14304 #J/(kg*K), specific heat capacity at constant pressure (H2)
C_p2 = 4187 #J/(kg*K)

#geometry
L=0.1            #m, tube length
R=0.0075         #m, tube radius



#species properties (H2 and H20)
sigma_A = 2.915;               #(Angstrom), collision diameter species A (H2)
sigma_B = 2.725;               #(Angstrom), collision diameter species B (H20)
sigma_AB = (sigma_A+sigma_B)/2 #average collision diameter
epsA = 38     #K   
epsB = 361.53 #K

#butler volmer
c_w = p/(R_gas*T_w) #density wall mol/m^3     
j0 = 1.2e4
alpha = 0.5   #transfer coefficients (anode=cathode)
eta = 0.1 #constant for now
beta1 = 1
n=2
k_boltz = 1.380649e-23;  # {J/K} Boltzmann Constant
# Re = rho*uAVG*2*R/mu #Reynolds number, radius based

def diffusivity(T):
    BE = T/np.sqrt(epsA * epsB);                                                         # Binary Energy Ratio
    omega_AB_0 = (0.4609*BE**2 + 33.2*BE + 37.6)/(BE**2 + 44.96*BE + 3.747);                 # Collision Integral
    D = 1e-4 * 0.0018583*(T**3*(1/M_A+1/M_B))**0.5 /(p_atm*sigma_AB**2*omega_AB_0) #reference diffusion coeff. based on reference temperature
    return D

D_ref = diffusivity(T_0) #reference diffusivity (inlet)
D_w = diffusivity(T_w) #wall diffusivity

#density
def rho_mix(T,p,M_mix):
        return p*M_mix*1e-3/(R_gas*T)

#specific heat
def C_p_mix(X1,X2):
    return X1*C_p1 + X2*C_p2

#dimensionless quantities
a = k/(rho_mix(T_0,p,M_A)*C_p1) #thermal diffusivity (Hydrogen only simplification)
Pe_L = uAVG*L/a #Peclet number
Pe_iL = uAVG*L/D_ref #Species Peclet number
pi_p = p-p_0/(p_0-p_e)
L_R = L/R
R_L = R/L

#--High Level Operational Controls--
#Collocation grid initial
boundary_pts = 1500
interior_pts = 2000

#Adam Solver
iter_Adam_initialize = 5000 #iterations of Adam before RAR

#Residual-based Adaptive Refinement
iter_Adam_RAR = 1000 #iterations of Adam between RAR steps
max_run_RAR = 0 #steps of RAR 
gtol_RAR = 1e-7 #Residual to stop RAR 
pts_add = 100

#LBFGS Solver
maxiter_LBFGS = 2000 #max iterations to run LBFGS
gtol_LBFGS = 1e-8 
#------------------------------------------

#2) FORMAT GEOMETRY

flowbox = dde.geometry.Rectangle([0,0],[1,1])
# timedomain = dde.geometry.TimeDomain(0, 10)

# geom = dde.geometry.GeometryXTime(flowbox, timedomain)
geom = flowbox #allows for addition of other geometry modifications


#3) Compute Physics Equations
def physLoss(x,y):
    """
    x: Network inputs [z,r] (Coordinates)
    y: Network outputs [X1,T] (species 1 (H2) and temperature)
    """

    #---grab inputs and outputs---
    z = x[:,0:1] #pull axial position
    r = x[:,1:2] #pull radial position
    
    #Analytical fluid flow solution
    u_r = 0 
    u_z = 2*(1-r**2) #nondimensional (/Uavg)

    
    #p =64/Re*(L-z)/(4*R)*rho*uAVG**2 #gage pressure relative to exit of tube

    X1 = y[:,0:1] #Mole fraction species 1, H2 (used)
    X2 = 1-X1   #Mole fraction species 2, H20 (produced)
    T = y[:,1:2] #Temperature out, non-dimensional


    #re-dimensionalize T for diffusion calc
    T_corr = T_0 + T*(T_w-T_0)

    M_mix = X1*M_A + X2*M_B #average molar mass
    M_mix_bar = M_mix/M_0
    M_A_bar = M_A/M_0
    M_B_bar = M_B/M_0
    Psi_1 = X1/M_mix_bar 
    Psi_2 = X2/M_mix_bar

    #density
    rho_bar = rho_mix(T_corr,p,M_mix)/rho_1 #rho_1 is the reference density

    #specific heat
    C_p_mix_bar = C_p_mix(X1,X2)/C_p1
    C_p1_bar = C_p1/C_p1
    C_p2_bar = C_p2/C_p1


    #Diffusion calculations
    D = diffusivity(T_corr)
    # D = D_ref #uncomment to turn off temperature coupling
    D_bar = D/D_ref
    # porosity = 0.2 #of anode
    # tort = 1.5 #tortuosity of anode
    # D_eff = D*porosity/tort #effective diffusion coefficient through combined effects of fluid and porous solid diffusion


    #---Compute All Derivatives (non-dimensional)---
    #FIRST DERIVATIVES
    #fluid
    du_z_z = 0
    du_r_r = 0 
    du_r_z = 0
    du_z_r = -4*r/R**2 #non-dimensional (/uAVG)

    #species
    dX1_z = dde.grad.jacobian(y,x,i=0,j=0)
    dX1_r = dde.grad.jacobian(y,x,i=0,j=1)
    dPsi1_z = dde.grad.jacobian(Psi_1,x,i=0,j=0)
    dPsi1_r = dde.grad.jacobian(Psi_1,x,i=0,j=1)
    dPsi2_z = dde.grad.jacobian(Psi_2,x,i=0,j=0)
    dPsi2_r = dde.grad.jacobian(Psi_2,x,i=0,j=1)

    #diffusion
    dD_z = dde.grad.jacobian(D_bar,x,i=0,j=0) 
    dD_r = dde.grad.jacobian(D_bar,x,i=0,j=1) 
    # dD_z = 0 #uncomment to overwrite temperture coupling
    # dD_r = 0 #uncomment to overwrite temperture coupling
   
    #temperature
    dT_z = dde.grad.jacobian(y,x,i=1,j=0)
    dT_r = dde.grad.jacobian(y,x,i=1,j=1)

    # density
    drho_z = dde.grad.jacobian(rho_bar,x,i=0,j=0)
    drho_r = dde.grad.jacobian(rho_bar,x,i=0,j=1)

    #pressure
    dp_z = 0 #constant pressure
    dp_r = 0 #constant pressure
  

    #SECOND DERIVATIVES
    #species
    dX1_zz = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dX1_rr = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dPsi1_zz = dde.grad.hessian(Psi_1, x, component=0, i=0, j=0)
    dPsi1_rr = dde.grad.hessian(Psi_1, x, component=0, i=1, j=1)
   

    #temperature
    dT_zz = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dT_rr = dde.grad.hessian(y, x, component=1, i=1, j=1)
  
    

    #---Compute Losses--- 
    #Species transport and energy balance
    #No time transients
    #variables in following equations are assumed to be dimensionless, so additional notation is not applied 
  
    # lossSpec = r/R_L *u_r*dPsi1_r + r*u_z*dPsi1_z - 1/Pe_iL*(r*D_bar/(R_L**2)*dPsi1_rr + D_bar/(R_L**2)*dPsi1_r + r/R_L**2*dD_r*dPsi1_r + r*D_bar*dPsi1_zz + r*dD_z*dPsi1_z)
    lossSpec = r*u_r/R_L*dPsi1_r + r*u_z*dPsi1_z - 1/Pe_iL*(r*D_bar/R_L**2*dPsi1_rr + D_bar/(R_L**2)*dPsi1_r + r*D_bar*dPsi1_zz + r/R_L**2*dD_r*dPsi1_r + r*dD_z*dPsi1_z + r*D_bar/(R_L**2*rho_bar)*drho_r*dPsi1_r + r*D_bar/rho_bar*drho_z*dPsi1_z)

    # lossEnergy = r*u_r/R_L*dT_r + r*u_z*dT_z - 1/Pe_L*(1/R_L**2 *(r*dT_rr + dT_r) + r*dT_zz)
    lossEnergy = r*rho_bar*(u_r/R_L*dT_r+u_z*dT_z) - 1/Pe_L*(r/R_L**2*dT_rr + 1/(R_L**2)*dT_r + r*dT_zz) - r*pi_p*(u_r/R_L*dp_r + u_z*dp_z) + r*C_p1_bar*((-rho_bar*D_bar*M_A_bar*dPsi1_r/(R_L**2*Pe_iL))*dT_r - (rho_bar*D_bar*M_A_bar*dPsi1_z/Pe_iL)*dT_z) + r*C_p2_bar*((-rho_bar*D_bar*M_B_bar*dPsi2_r/(R_L**2*Pe_iL))*dT_r - (rho_bar*D_bar*M_B_bar*dPsi2_z/Pe_iL)*dT_z)
    return [lossSpec, lossEnergy]

#4) Boundary Conditions

#Define Boundary Condition Locations
def top_wall(x, on_boundary):
    # Adjust 'y_max_coordinate' to match your actual geometry height
    return (on_boundary and np.isclose(x[1], 1) and not np.isclose(x[0], 0)) #also removes z=0 from wall to prevent X1 value conflict with inlet

def bottom_wall(x, on_boundary):
    # Adjust 'y_min_coordinate' to match your actual geometry base
    return on_boundary & np.isclose(x[1], 0)

def left_wall(x, on_boundary):
    # Adjust 'x_min_coordinate' to match your actual geometry left edge
    return on_boundary & np.isclose(x[0], 0)

def right_wall(x, on_boundary):
    # Adjust 'x_min_coordinate' to match your actual geometry right edge
    return on_boundary & np.isclose(x[0], 1)

def wall_func(x, y,_):
  # x is a NumPy array of shape (N, 1) on the boundary
  X1 = y[:,0:1]
  n=2
  X1_0 = 1
  j = j0*(np.exp(n*alpha*F*eta/(R_gas*T_w))-np.exp(-n*(1-alpha)*F*eta/(R_gas*T_w)))*((X1/X1_0)**beta1) 
  #j = 1.2e6*X1
  J = -j/(2*F)
  return J - (c_w*D_w/R)*dde.grad.jacobian(y, x, i=0, j=1)


#Associate boundary conditions with the walls

# --- UPPER WALL:  ---
bc_wall_X1 = dde.icbc.OperatorBC(geom, wall_func, top_wall,)
bc_wall_T = dde.icbc.DirichletBC(geom, lambda x: 1, top_wall, component=1)

# --- LOWER WALL (AXIS): ---
bc_axis_X1 = dde.icbc.OperatorBC(geom, lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1), bottom_wall,)
bc_axis_T = dde.icbc.OperatorBC(geom, lambda x, y, _: dde.grad.jacobian(y, x, i=1, j=1), bottom_wall,)

# --- LEFT WALL (Inlet):  ---
bc_left_X1 = dde.icbc.DirichletBC(geom, lambda x: 1, left_wall, component=0)
bc_left_T = dde.icbc.DirichletBC(geom, lambda x: 0, left_wall, component=1)

# --- RIGHT WALL (Outlet): ---
bc_right_X1 = dde.icbc.OperatorBC(geom, lambda x, y, _: (c_w*D_w/L)*dde.grad.jacobian(y, x, i=0, j=0), right_wall,) #assuming D_w at exit due to close temperatures  #FIX RHO, NOT CONSTANT
bc_right_T = dde.icbc.OperatorBC(geom, lambda x, y, _: dde.grad.jacobian(y, x, i=1, j=0),right_wall,)

#Combine boundary conditions
# bc_X1 = [bc_wall_X1, bc_axis_X1, bc_left_X1, bc_right_X1] 
# bc_T = [bc_wall_T, bc_axis_T, bc_left_T]
# bcs = [bc_X1, bc_T] #combine

bcs = [bc_wall_X1, bc_axis_X1, bc_left_X1, bc_right_X1, bc_wall_T, bc_axis_T, bc_left_T, bc_right_T] 

# 5). Combine everything into a DeepXDE Data Object
data = dde.data.PDE(
    geom,
    physLoss,
    bcs,            
    num_domain=interior_pts,  # Number of collocation points inside the domain
    num_boundary=boundary_pts, # Number of points on the boundary
)

# 6). Construct the Neural Network Architecture
layer_size = [2] + [20] * 5 + [2]  # Input (x, t) -> 5 hidden layers of 20 neurons -> Output (u_z, u_r, p)
activation = "tanh"
initializer = "Glorot normal"

#Set early stopping criteria
early_stopping = dde.callbacks.EarlyStopping(
    monitor="loss_train",       # Options: 'loss_train' or 'loss_test'
    baseline=1e-3,              # Stops training once total loss <= baseline
    patience=0                  # Stop immediately without waiting for extra epochs
)

dde.optimizers.config.set_LBFGS_options(
    maxcor=3000,
    ftol=0,
    gtol=gtol_LBFGS,
    maxiter=maxiter_LBFGS,
    maxfun=None,
    maxls=50
)

net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)


# 7). Start Training the Model W/ Adam

# [LossSpec, LossT, BCwallX1, BCaxisX1, BCinletX1, BCoutletX1 , BCwallT, BCaxisT, BCinletT]
weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

print("Training with Adam optimizer...")
model.compile("adam", lr=0.001, loss_weights=weights)


#---Plot current collocation grid---
x_train = data.train_x 

x_coords = x_train[:, 0]
y_coords = x_train[:, 1]

# plt.figure(figsize=(6, 6))
# plt.scatter(x_coords, y_coords, s=15, c='blue', alpha=0.7, label='Collocation Points')
# plt.title("DeepXDE Collocation Grid")
# plt.xlabel("X coordinate")
# plt.ylabel("Y coordinate")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend()
# plt.show(block=False)
# plt.pause(0.001)
#------------------------------



training_log = TrainingLog(model)  # records loss history and training time
with training_log.stage("Adam (initial)"):
    model.train(iterations=iter_Adam_initialize) #Adam iterations before RAR

# 8). Residual-based Adaptive Refinement W/ Adam

max_rar_iterations = max_run_RAR
points_to_add_per_iteration = pts_add
residual_threshold = gtol_RAR

for i in range(max_rar_iterations):
    
    
    # 1. Sample a large pool of random candidate points across the domain
    X_candidates = geom.random_points(interior_pts) #define number points to sample (interior collocation points)
    
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
    training_log.add_rar_points(new_points, rar_round=i + 1)
    
    # 8. Retrain the network with the newly added constraints
    with training_log.stage(f"Adam (RAR {i + 1})"):
        model.train(iterations=iter_Adam_RAR) #define how many training iterations between RAR sessions 




#---Plot collocation grid after RAR--- 
# x_train = data.train_x 

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


#9) L-BFGS optimizer to finely tune the physics losses
print("Fine-tuning with L-BFGS optimizer...")
model.compile("L-BFGS", loss_weights=weights)
with training_log.stage("L-BFGS"):
    model.train()


# Save trained model to models/v3/ folder
model_dir = model_directory("V3")
os.makedirs(model_dir, exist_ok=True)

# Name includes model version and training date/time (DeepXDE appends -<step>.pt)
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
save_path = model.save(f"{model_dir}/pinn_model_V3_{timestamp}")

# Prints exact file name for saved model
print("Saved model: ")
print(save_path)

# Save loss history and training time next to the model, then save its plots
training_log.save(save_path)
plot_after_training(save_path)



#10) Predict and Visualize Results

# Generate a grid of points to evaluate the trained model
x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_vals, y_vals)
points = np.vstack((np.ravel(X), np.ravel(Y))).T

# Make predictions (returns an array of [u, v, p])
predictions = model.predict(points)

X1 = predictions[:, 0].reshape(100, 100)
T = T_0 + (T_w-T_0)*predictions[:, 1].reshape(100, 100)


# ---Plotting X1 contour--- 
plt.figure(figsize=(10, 2))
contour = plt.contourf(X, Y, X1, levels=100, cmap='jet')

# Create the colorbar
cbar = plt.colorbar(contour, label='Molar Fraction')

# Apply formatting: '%g' removes scientific notation, and '.f' controls decimals
# For example, precision=4 becomes '%.4f'
formatter = ticker.FormatStrFormatter(f'%.4f')
cbar.ax.yaxis.set_major_formatter(formatter)

plt.title('PINN Prediction: H2 Contour In Tube')
plt.xlabel('$x$ (Length)')
plt.ylabel('$y$ (Height)')
plt.tight_layout()
plt.show(block=False)

# ---Plotting T contour--- 
plt.figure(figsize=(10, 2))
contour = plt.contourf(X, Y, T, levels=100, cmap='jet')

# Create the colorbar
cbar = plt.colorbar(contour, label='Temperature')

# Apply formatting: '%g' removes scientific notation, and '.f' controls decimals
# For example, precision=4 becomes '%.4f'
formatter = ticker.FormatStrFormatter(f'%.4f')
cbar.ax.yaxis.set_major_formatter(formatter)

plt.title('PINN Prediction: T Contour in Tube')
plt.xlabel('$x$ (Length)')
plt.ylabel('$y$ (Height)')
plt.tight_layout()
plt.show(block=False)



#--- Plotting X1 Along Wall ---
z_hat = np.linspace(0, 1, 200)

wall_points = np.column_stack((
    z_hat,                         # Axial coordinate: z/L
    np.full_like(z_hat, 1)       
))

X1_wall = model.predict(wall_points)[:, 0]

z_wall = L * z_hat  # Convert to physical axial position, meters

plt.figure(figsize=(8, 4))
plt.plot(z_wall, X1_wall, label=r"$H_2$")
plt.title('XH2 Mole Fraction Wall')
plt.xlabel("Axial position z (m)")
plt.ylabel("Mole Fraction H2")
plt.show()

#---CALCULATE AVERAGE SPECIES AT OUTLET PLANE---
r_hat = np.linspace(0, 1, 100)

outlet_points = np.column_stack((
    np.ones_like(r_hat),  # z_hat = 1 at the outlet
    r_hat
))

X1_out = model.predict(outlet_points)[:, 0]

X1_area_avg = (
    np.trapezoid(X1_out * r_hat, r_hat)
    / np.trapezoid(r_hat, r_hat)
)

print("Outlet area-average X1 =", X1_area_avg)

# #---Generate total loss heat map---
# # Grid: inputs are [z/R, r/R]
# z_vals = np.linspace(0, 5, 200)
# r_vals = np.linspace(0, 1, 100)
# Z, RR = np.meshgrid(z_vals, r_vals)
# points = np.column_stack((Z.ravel(), RR.ravel()))

# # Evaluate the three PDE residuals using automatic differentiation
# residuals = model.predict(points, operator=physLoss)
# residuals = np.hstack(residuals)  # Shape: (number of points, 3)

# if not np.isfinite(residuals).all():
#     raise ValueError(
#         "PDE residuals contain NaN or infinity. Check the axis treatment."
#     )

# # Pointwise sum of squared PDE residuals
# local_loss = np.sum(residuals.astype(np.float64)**2, axis=1)
# local_loss = local_loss.reshape(Z.shape)

# fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)

# heatmap = ax.pcolormesh(
#     R * Z,
#     R * RR,
#     np.maximum(local_loss, 1e-20),
#     shading="auto",
#     cmap="inferno",
#     norm=LogNorm(),)

# fig.colorbar(heatmap, ax=ax, label="Sum of squared PDE residuals")
# ax.set_xlabel("Axial position z (m)")
# ax.set_ylabel("Radius r (m)")
# ax.set_title("Local PDE residual")
# plt.show()


