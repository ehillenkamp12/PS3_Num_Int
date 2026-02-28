import numpy as np
import matplotlib.pyplot as plt
import scipy

# Turn on interactive mode and set up plot
plt.ion()
fig, ax = plt.subplots()

# Viscous diffusion coefficient 
nu = 1

# Set up spatial array and sigma (surface density)
r = np.arange(0.005,10,.05)

# Advection velocity 
u = -(9/2)*nu/np.copy(r)
u[0] = -abs(u[1])
u[-1] = -abs(u[-2])

# Diffusion coefficient 
D = 3*nu

# Using Lax-Friedrichs to solve the advection term 
# Start s as a Gaussian with stddev=0.005 (sharply peaked), centered at x=1
s = (1. / (0.05*np.sqrt(2*np.pi))) * np.exp(-0.5*((np.copy(r)-3)/.05)**2) 

# Time array 
t = np.arange(0,10,.0001) # Courant condition -- dt <= dx/|u|

# Advection velocity coefficient array
a = u*(t[1]-t[0]) / (2*(r[1]-r[0]))

# Diffusion coefficient 
B = D*(t[1]-t[0])/((r[1]-r[0])**2)
n = len(r) # Length of spatial array
# Create tri-diagonal matrix with -B, 2B+1, and -B on the diagonals for implicit method
A = np.eye(n, k=1) * (-B) + np.eye(n) * (2.0*B + 1.0) + np.eye(n, k=-1) * (-B)

# Set axis limits and labels
ax.set(ylim=(-1,10),xlim=(.005,10), xscale='log')
ax.set(ylabel='$\Sigma$',xlabel='r',title='Accretion Disk Numerical Integration Animation')

# Plot initial values as line and starting point
ax.plot(r, s, linestyle='-', color='r', label='Initial Profile') # Fixed
vals, = ax.plot(r, s, color='b', label='Evolution of Surface Density') # Will update with time 
plt.legend()


# For each time step, update s with diffusion term and then advection term, update plot
for t_step in t:
    s = s[np.newaxis,:].T # Reshape s into column vector
    
    # Diffusion
    s[:] = np.linalg.solve(A,s)
    s = s.ravel() # Back into array
    
    # Advection
    # s at each time step is updated from the previous step, using adjacent values of s
    s[1:-1] = 0.5*(s[2:]+s[0:-2]) + (a[1:-1]*(s[2:]-s[0:-2])) # Lax-Friedrichs 
    # Boundary conditions 
    s[0] = s[1]
    s[-1] = s[-2]

    # Update plot
    vals.set_ydata(s)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # Animation cadence
    plt.pause(.000001)
    
plt.close()