import numpy as np
import matplotlib.pyplot as plt
import estimators

# Folder to save images in
save_dir = '../Images/'

# Seed
np.random.seed(10)

# Option to either show or save the image
save_fig = True

# Create plot 1: Linear
f_original = lambda x: x # Non-noisy function
sigma = 0.5 # Noise (standard deviation of gaussian noise)
f_noisy = lambda x: f_original(x) + np.random.normal(0,sigma,x.shape) # Non-noisy function

x_min = -5
x_max = 5

num_pts_original = 1000 # Points for non-noisy function
num_pts_noisy = 100 # Noisy points

x_eval_original = np.linspace(x_min,x_max,num_pts_original)
x_eval_noisy = np.linspace(x_min,x_max,num_pts_noisy)

y_eval_original = f_original(x_eval_original)
y_eval_noisy = f_noisy(x_eval_noisy)

plt.plot(x_eval_original,y_eval_original,color='black',label="Original Function")
plt.scatter(x_eval_noisy,y_eval_noisy,s=5,color='red',label="Noisy Function")
plt.legend()

if save_fig:
    plt.savefig(save_dir + "intro_1.pdf")
else:
    plt.show()
plt.close()

# Create plot 2: Cubic
f_original = lambda x: x**3 # Non-noisy function
sigma = 10 # Noise (standard deviation of gaussian noise)
f_noisy = lambda x: f_original(x) + np.random.normal(0,sigma,x.shape) # Non-noisy function
f_quad = lambda x: -1*6*(x+0.5)**2

x_min = -5
x_max = 5

num_pts_original = 1000 # Points for non-noisy function
num_pts_noisy = 5 # Noisy points

x_eval_original = np.linspace(x_min,x_max,num_pts_original)
x_eval_noisy = np.random.uniform(x_min,x_max,num_pts_noisy)

y_eval_original = f_original(x_eval_original)
y_eval_noisy = f_noisy(x_eval_noisy)

R2 = estimators.ridge(gamma=0, degree=2)

R2.fit(x_eval_noisy,y_eval_noisy)
y_eval_d2 = R2.predict(x_eval_original)

plt.plot(x_eval_original,y_eval_original,color='black',label="Original Function")
plt.plot(x_eval_original,y_eval_d2,color='blue',linestyle='dotted',label="Degree 2 fit")
plt.scatter(x_eval_noisy,y_eval_noisy,s=5,color='red',label="Noisy Function")
plt.ylim(-150,150)
plt.legend()

if save_fig:
    plt.savefig(save_dir + "intro_2.pdf")
else:
    plt.show()
plt.close()

# Create plot 3: Cubic
f_original = lambda x: x**3 # Non-noisy function
sigma = 10 # Noise (standard deviation of gaussian noise)
f_noisy = lambda x: f_original(x) + np.random.normal(0,sigma,x.shape) # Non-noisy function
#f_quad = lambda x: -1*6*(x+0.5)**2

x_min = -5
x_max = 5

num_pts_original = 100 # Points for non-noisy function
num_pts_noisy = 5 # Noisy points

x_eval_original = np.linspace(x_min,x_max,num_pts_original)
x_eval_noisy = np.random.uniform(x_min,x_max,num_pts_noisy)

deg = 4
R2 = estimators.ridge(gamma=0, degree=deg)

y_eval_original = f_original(x_eval_original)
y_eval_noisy = f_noisy(x_eval_noisy)

R2.fit(x_eval_noisy,y_eval_noisy)
y_eval_d2 = R2.predict(x_eval_original)

plt.plot(x_eval_original,y_eval_original,color='black',label="Original Function")
plt.scatter(x_eval_noisy,y_eval_noisy,s=5,color='red',label="Noisy Function")
plt.plot(x_eval_original,y_eval_d2,color='blue',linestyle='dotted',label=f"Degree {deg} fit")
plt.ylim(-150,150)
plt.legend()

if save_fig:
    plt.savefig(save_dir + "intro_3.pdf")
else:
    plt.show()
plt.close()