import numpy as np
import random as random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from matplotlib.animation import PillowWriter


#This code is adapted from my homework exercise I did in my Lattice Field Theory course
#The course in question is https://github.com/KariRummukainen/Lattice-Field-Theory
#I took this course in early 2024
#I have added visualisation aspects for the quantum fields and modified the simulation functions from my code for the exercise.

#The simulated field is a field with the following action
# S_L =  \sum_x \bigg[ - 2\kappa \sum_\mu \phi_x \phi_{x+\mu} + (1-2\lambda) \phi^2 + \lambda \phi^4 \bigg]



#Some of the functions are taken from the course GitHub and then modified for this exercise



def computeMagnetization(field):
    return np.abs(np.mean(field))


def updateMetropolisHastings(lattice_size, kappa, lamb, field): 

    x = int( lattice_size*np.random.random() )
    y = int( lattice_size*np.random.random() )


    old_field = field[x][y]
    new_field = old_field + np.random.standard_normal()


    S_old = -2 * kappa * old_field*( field[x][(y+1)%lattice_size] + field[x][y-1] + field[(x+1)%lattice_size][y] + field[x-1][y]) + (1-2*lamb)*(old_field**2) + lamb*(old_field**4)
    
    S_new = -2 * kappa * new_field*( field[x][(y+1)%lattice_size] + field[x][y-1] + field[(x+1)%lattice_size][y] + field[x-1][y]) + (1-2*lamb)*(new_field**2) + lamb*(new_field**4)

    #Metropolis Hasting
    dS = S_new - S_old
    if dS <= 0.0:
        #Accept!
        field[x][y] = new_field
    else:
        #P_flip just the probability to accept new field value (naming carried on from the GitHub example)
        P_flip = np.exp(-dS)
        if np.random.random() < P_flip:
            field[x][y] = new_field
    return field

def simulation(lattice_size, burn_in, steps, kappa, lamb, field, skip=1):
    field = field
    M = 0.0
    num_measure = 0

    for b in range(burn_in):
        field = updateMetropolisHastings(lattice_size, kappa, lamb, field)
    
    #Lattice_states snaptshots the lattice at every step
    lattice_states = []

    for b in range(steps):
        field = updateMetropolisHastings(lattice_size, kappa, lamb, field)
        
        if b % skip == 0 :
            M += computeMagnetization(field)
            num_measure += 1
            #print(kappa, b) #For debugging purposes
        
        lattice_states.append(np.copy(field))

    absolute_magnetization = M/num_measure
    return lattice_states, field, absolute_magnetization

def interpolate_field(Z, lattice_size):
    x = np.arange(lattice_size)
    y = np.arange(lattice_size)
    X, Y = np.meshgrid(x, y)

    # Flatten the data for interpolation
    points = np.array([X.flatten(), Y.flatten()]).T
    values = Z.flatten()

    # Interpolation grid
    finer_x = np.linspace(0, lattice_size - 1, lattice_size * 4)
    finer_y = np.linspace(0, lattice_size - 1, lattice_size * 4)
    finer_X, finer_Y = np.meshgrid(finer_x, finer_y)

    # Interpolate to a finer grid
    finer_Z = griddata(points, values, (finer_X, finer_Y), method='cubic')

    return finer_X, finer_Y, finer_Z

def magnetization_simulation(lattice_size=16, burn_in_coef=10000, kappa_range=[0.22, 0.3, 12], steps=50, lamb=0.02):
    burn_in = 10000 * (lattice_size**2)
    kappa = np.linspace(kappa_range[0], kappa_range[1], kappa_range[2])

    MT = []

    for k in kappa:
        field = np.random.rand(lattice_size, lattice_size)
        _, field, M = simulation(lattice_size, burn_in, steps, k, lamb, field, skip=2)
        MT.append(M)
    
    plt.figure()
    plt.plot(kappa, MT)
    plt.title(r"Absolute magnetization as function of $\kappa$")
    plt.xlabel(r'$\kappa$')
    plt.ylabel(r'$|\langle M \rangle |$')
    plt.savefig('magnetization.pdf')
    plt.show()

def simulation_of_fields_animation(lattice_size=14, burn_in_coef=10000, steps=1000, kappa=0.25, lamb=0.02, style=2):
    """
    style : int, optional (default=2)
        The style of visualization:
        - 2: 2D visualization (default)
        - 3: 3D visualizatio
    """

    burn_in = burn_in_coef * (lattice_size**2)


    #initialize the field
    field = np.random.rand(lattice_size, lattice_size)

    #The simulation with the snapshots being 
    lattice_states, _, _ = simulation(lattice_size, burn_in, steps, kappa, lamb, field)


    if style==3:
        #This code is for 3D visualization

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        X, Y = np.meshgrid(np.arange(lattice_size), np.arange(lattice_size))
        Z = lattice_states[0] #Initial field values after burn in (amplitudes)

        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', vmin=-4, vmax=4)

        ax.set_title("Lattice field evolution")
        ax.set_xlabel("X")
        ax.set_ylabel('Y')
        ax.set_zlabel('Field Value (Amplitude)')

        #This bit is for an interpolated field, this makes the fields look much smoother in a small lattice
        #Turn this off by commenting the update function and removing the comment on the one below.
        def update(frame):
            nonlocal surf
            surf.remove()

            Z = lattice_states[frame]
            finer_X, finer_Y, finer_Z = interpolate_field(Z, lattice_size)

            # Plot the interpolated surface
            surf = ax.plot_surface(finer_X, finer_Y, finer_Z, cmap='coolwarm', vmin=-4, vmax=4)
            ax.set_title(f"Lattice field evolution: Step {frame + 1}")
            return surf,

        #For a update without any interpolation:
        """
        def update(frame):
            nonlocal surf
            Z = lattice_states[frame]
            surf.remove()
            surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', vmin=-4, vmax=4)
            ax.set_title(f"Lattice field evolution: Step {frame + 1}")
            return surf,
        """

        # Animation stuff:
        ani = FuncAnimation(fig, update, frames=len(lattice_states), interval=100, blit=False)

        writer3d = PillowWriter(fps=30, metadata={'artist': 'Shankar'}, bitrate=1800)  # MP4 Writer
        ani.save("lattice_simulation3dgif.gif", writer=writer3d)
        print("Animation saved successfully!")

        plt.show()
    else:

        fig, ax = plt.subplots()
        img = ax.imshow(lattice_states[0], cmap="coolwarm", vmin=-4, vmax=4)
        ax.set_title("Lattice field evolution")
        plt.colorbar(img, ax=ax, label="Value of field")

        def update(frame):
            img.set_data(lattice_states[frame])
            ax.set_title(f"Lattice field evolution: Step {frame + 1}")
            return img,

        # Animation stuff:
        ani = FuncAnimation(fig, update, frames=len(lattice_states), interval=100, blit=False)
        writer2d = PillowWriter(fps=30, metadata={'artist': 'Shankar'}, bitrate=1800)  # MP4 Writer
        ani.save("lattice_simulation_2dgif.gif", writer=writer2d)  

        plt.show()

def main():
    simulation_of_fields_animation(style=3) #Simulation of fields with default parameters except for style which. style=3 gives us 3d visualisation

    magnetization_simulation(lattice_size=20) #Simulation of kappa dependence of Magnetization


main()





