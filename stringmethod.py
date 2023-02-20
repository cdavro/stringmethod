# @author Akash Pallath
# Licensed under the MIT license, see LICENSE for details.
#
# Reference: Weinan E, "Simplified and improved string method for computing the minimum energy paths in barrier-crossing events",
# J. Chem. Phys. 126, 164103 (2007), https://doi.org/10.1063/1.2720838

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata, interp1d, UnivariateSpline
from tqdm import tqdm


class String2D:
    """
    Class containing methods to compute the minimum energy path between two
    points on an energy landscape $V$.

    Args:
        x: Array of shape (nx,) specifying x-axis coordinates of grid.
        y: Array of shape (ny,) specifying y-axis coordinates of grid.
        V: Array of shape (ny, nx) or (nx, ny) specifying energy values at each point on the grid.
            Missing values should be set to np.inf.
        indexing: Indexing of V array ('xy' specifies (ny, nx), 'ij' specifies (nx, ny); default = 'xy').

    Attributes:
        x: Array of shape (nx,) specifying x-axis coordinates of grid.
        y: Array of shape (ny,) specifying y-axis coordinates of grid.
        V: Array of shape (ny, nx) or (nx, ny) specifying energy values at each point on the grid.
        X: Grid of shape (ny, nx) or (nx, ny) containing x-coordinates of each point on the grid.
        Y: Grid of shape (ny, nx) or (nx, ny) containing y-coordinates of each point on the grid.
        indexing: Indexing of V, X, and Y arrays ('xy' specifies (ny, nx), 'ij' specifies (nx, ny); default = 'xy').
        gradX: Gradient in x.
        gradY: Gradient in y.
        string_traj: Trajectory showing the evolution of the string (default=[]).
        mep: Converged minimum energy path (default=None, if not converged).
    """
    def __init__(self, x, y, V, indexing='xy'):
        self.x = x
        self.y = y
        self.V = V

        # Generate grids
        self.X, self.Y = np.meshgrid(x, y, indexing=indexing)
        self.grid = np.vstack([self.X.ravel(), self.Y.ravel()]).T

        # Compute gradients
        self.indexing = indexing

        if self.indexing == 'xy':
            self.gradY, self.gradX = np.gradient(self.V, self.x, self.y)
        elif self.indexing == 'ij':
            self.gradX, self.gradY = np.gradient(self.V, self.x, self.y)
        else:
            raise ValueError("Indexing method not recognized.")

        # String method variables
        self.string_traj = []
        self.mep = None

    def compute_mep(self, in_mep=[], begin=[], end=[], mid=[], spline_order=0, npts=100, integrator='forward_euler', dt=0.1, tol=None, maxsteps=100, traj_every=10, print_conv=False):
        """
        Computes the minimum free energy path. 
        If `in_mep` is provided as a path, use it as a guess (no need for begin/end/mid)
        The points `begin`
        and `end` and the midpoints passed through `mid` are used to generate
        an initial guess (a k-order spline which interpolates through all the points).
        If spline_order  0, then lines connecting the segments with a number of points in-between depending on the euclidean norm between points.
        If no midpoints are defined, then the initial guess is a line connecting `begin`
        and `end`. The ends of the string are free to move, and are not fixed to `begin`
        and `end`.

        Args:
            in_mep: An array of shape (npts_input,2) specifying string images (default=[]).
            begin: Array of shape (2,) specifying starting point of the string (default=[]).
            end: Array of shape (2,) specifying end point of the string (default=[]).
            mid: List of arrays of shape (2,) specifying points between `begin` and `end`
                to use for generating an initial guess of the minimum energy path (default=[]).
            spline_order: Order of spline interpolating begin, mid, and end (default=0) [if 0, linear segements betweens points].
            npts: Number of points between any two valuesalong the string (default=100).
            integrator: Integration scheme to use (default='forward_euler'). Options=['forward_euler','runge_kutta_4'].
            dt: Integration timestep (default=0.1).
            tol: Convergence criterion; stop stepping if string has an RMSD < tol between
                consecutive steps (default = max{npts^-4, 10^-10}).
            maxsteps: Maximum number of steps to take (default=100).
            traj_every: Interval to store string trajectory (default=10).
            print_conv: Print convergence values every 'traj_every' (default=False).

        Returns:
            mep: Array of shape (npts, 2) specifying string images along the minimum energy path between `begin` and `end`.
        """

        # Calculate params
        if tol is None:
            tol = max([npts ** -4, 1e-10])

        # Generate initial guess
        npts_in = np.asarray(in_mep).shape[0]
        
        # Check if a path is provided
        if npts_in > 1 :

            # If its length is equal to the number of points requested, then done
            if npts_in == npts:
                string = in_mep
                if np.asarray(string).shape[0] != npts: raise ValueError("Size of string isn't equal to number of points requested. This shouldn't happen, please report.")

            # If its length is superior to the number of points requested, then cut some points (as linearly as possible, keep the first and last points)
            elif npts_in > npts :
                path = in_mep[np.round(np.linspace(0, npts_in-1, npts)).astype(int)]
                string = path
                if np.asarray(string).shape[0] != npts: raise ValueError("Size of string isn't equal to number of points requested. This shouldn't happen, please report.")
    
            # If its length is inferior to the number of points requested, add points alternatively at the begining and the end, then do equal arc length reparameterization
            elif npts_in < npts :
                string = in_mep
                add = 0
                while string.shape[0] != npts:
                    if add == 0:
                        string=np.vstack((string[0],string))
                        add = 1
                    else:
                        string=np.vstack((string,string[-1]))
                        add = 0
                arclength = np.hstack([0, np.cumsum(np.linalg.norm(string[1:] - string[:-1], axis=1))])
                arclength /= arclength[-1]
                reparam_x = interp1d(arclength, string[:, 0])
                reparam_y = interp1d(arclength, string[:, 1])
                gamma = np.linspace(0, 1, npts)
                string = np.vstack([reparam_x(gamma), reparam_y(gamma)]).T
                if np.asarray(string).shape[0] != npts: raise ValueError("Size of string isn't equal to number of points requested. This shouldn't happen, please report.")

        # If no path is provided, check if the begin and end are non-zero
        elif (len(begin) > 0) and (len(end) > 0):

            # If no midpoints, just linear between begin and end
            if len(mid) == 0:
                string_x = np.linspace(begin[0], end[0], npts)
                string_y = np.linspace(begin[1], end[1], npts)
                string = np.vstack([string_x, string_y]).T
                if np.asarray(string).shape[0] != npts: raise ValueError("Size of string isn't equal to number of points requested. This shouldn't happen, please report.")

            # If midpoints
            else:
            
                # If linear is requested
                if spline_order == 0:
                    
                    # If the number of points requested for the string is less than the number of points in input (begin/mid/end)
                    if (  npts < (len(mid) + 2) ) :
                        raise ValueError("The total number of points can't be lower than initial number of points provided (beg/mid/end)")
                    
                    # If the number of points requested for the string is equal to the number of points in input (begin/mid/end)
                    elif (len(mid) + 2) == npts:
                        string_x = np.hstack((begin[0], [mid[n][0] for n in range(0,len(mid))], end[0]))
                        string_y = np.hstack((begin[1], [mid[n][1] for n in range(0,len(mid))], end[1]))

                    # Calculate the (euclidean) norm between points, distribute points per segments function of length (longer, more points)
                    elif len(mid) >= 1:
                        npts_toadd = npts - ( len(mid) + 2 )

                        begin=np.array([begin])
                        end=np.array([end])
                        mid=np.array(mid)

                        in_path=np.vstack((begin, np.array([mid[n] for n in range(0,mid.shape[0])]), end))

                        segment_count =  ( mid.shape[0] + 2 ) - 1
                        segment_lenght=np.zeros(segment_count)
                        for n in range(0,segment_count):
                            segment_lenght[n] = np.linalg.norm(in_path[n+1,:]-in_path[n,:])

                        segment_n_lenght=segment_lenght/np.sum(segment_lenght)
                        segment_npts=np.round((npts_toadd*segment_n_lenght),0).astype(int)

                        # Because of rounding, points might be under or over
                        max_count,min_count=1,0
                        # If under, distribute points to larger distances (one point to the largest, second point to the second largest, etc..)  
                        while mid.shape[0] + 2 + np.sum(segment_npts) < npts:
                            segment_npts[np.where(np.sort(segment_npts)[-max_count])] += 1
                            max_count += 1
                        # If over, distribute points to smaller distances (minus one point to the smallest, minus second point to the second smallest, etc..) 
                        while mid.shape[0] + 2 + np.sum(segment_npts) > npts:
                            segment_npts[np.where(np.sort(segment_npts)[min_count])] -= 1
                            min_count += 1

                        string_x = np.linspace(in_path[0][0], in_path[1][0], segment_npts[0] + 2)
                        string_y = np.linspace(in_path[0][1], in_path[1][1], segment_npts[0] + 2)
                        for n in range(1,segment_count):
                            string_x = np.hstack((string_x, np.linspace(in_path[n][0],in_path[n+1][0], segment_npts[n]+ 2)[1:]))
                            string_y = np.hstack((string_y, np.linspace(in_path[n][1],in_path[n+1][1], segment_npts[n]+ 2)[1:]))

                    string = np.vstack([string_x, string_y]).T
                    if np.asarray(string).shape[0] != npts: raise ValueError("Size of string isn't equal to number of points requested. This shouldn't happen, please report.")

                # If spline is requested
                else:
                    if len(mid) > 0:
                        string_x = np.linspace(begin[0], end[0], npts)
                        xpts = [begin[0]] + [mpt[0] for mpt in mid] + [end[0]]
                        ypts = [begin[1]] + [mpt[1] for mpt in mid] + [end[1]]
                        spline = UnivariateSpline(xpts, ypts, k=spline_order)
                        string_y = spline(string_x)
                    string = np.vstack([string_x, string_y]).T
                    if np.asarray(string).shape[0] != npts: raise ValueError("Size of string isn't equal to number of points requested. This shouldn't happen, please report.")
        else:
            raise ValueError("If no path, it must have no-empty begin/end.")

        # Store initial guess
        self.string_traj = []
        self.string_traj.append(string)

        # Loop
        old_string = np.zeros_like(string)

        for tstep in tqdm(range(1, maxsteps + 1)):
            # Integrator step
            if integrator == "forward_euler":
                old_string[:] = string
                string = self.step_euler(string, dt)
            elif integrator == "runge_kutta_4":
                old_string[:] = string
                string = self.step_RK4(string, dt)
            else:
                raise ValueError("Invalid integrator")

            # Reparameterize string (equal arc length reparameterization)
            arclength = np.hstack([0, np.cumsum(np.linalg.norm(string[1:] - string[:-1], axis=1))])
            arclength /= arclength[-1]
            reparam_x = interp1d(arclength, string[:, 0])
            reparam_y = interp1d(arclength, string[:, 1])
            gamma = np.linspace(0, 1, npts)
            string = np.vstack([reparam_x(gamma), reparam_y(gamma)]).T

            # Store
            if tstep % traj_every == 0:
                self.string_traj.append(string)
                # Print convergence
                if print_conv == True:
                    print("Change in string: {:.10f}".format(np.sqrt(np.mean((string - old_string) ** 2))))

            # Test for convergence
            if np.sqrt(np.mean((string - old_string) ** 2)) < tol:
                break

        # Store minimum energy path
        self.mep = string

    def step_euler(self, string, dt):
        """
        Evolves string images in time in response to forces calculated from the energy landscape.

        Args:
            string: Array of shape (npts, 2) specifying string images at the previous timestep.
            dt: Timestep.

        Returns:
            newstring: Array of shape (npts, 2) specifying string images after a timestep.
        """
        # Compute gradients at string points
        string_grad_x = griddata(self.grid, self.gradX.ravel(), string, method='linear')
        string_grad_y = griddata(self.grid, self.gradY.ravel(), string, method='linear')
        h = np.max(np.sqrt(string_grad_x ** 2 + string_grad_y ** 2))

        # Euler step
        newstring = string - dt * np.vstack([string_grad_x, string_grad_y]).T / h

        return newstring

    def step_RK4(self, string, dt):
        """
        Evolves string images in time in response to forces calculated from the energy landscape.

        Args:
            string: Array of shape (npts, 2) specifying string images at the previous timestep.
            dt: Timestep.

        Returns:
            newstring: Array of shape (npts, 2) specifying string images after a timestep.
        """
        
        # Compute gradients at string points
        string_grad_x = griddata(self.grid, self.gradX.ravel(), string, method='linear')
        string_grad_y = griddata(self.grid, self.gradY.ravel(), string, method='linear')
        grad_norm = np.sqrt(string_grad_x ** 2 + string_grad_y ** 2)
        grad_norm[grad_norm == 0] = 1  # Avoid division by zero

        # Define the derivative function for the Runge-Kutta method
        def f(t, y):
            x, y = y.T
            grad_x = np.interp(x, self.grid[:, :, 0].ravel(), self.gradX.ravel())
            grad_y = np.interp(y, self.grid[:, :, 1].ravel(), self.gradY.ravel())
            return np.vstack([grad_x / grad_norm, grad_y / grad_norm]).T

        # Use the Runge-Kutta method to update the positions of the string points
        k1 = f(0, string)
        k2 = f(dt/2, string + dt/2 * k1)
        k3 = f(dt/2, string + dt/2 * k2)
        k4 = f(dt, string + dt * k3)
        newstring = string + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        return newstring

    def get_mep_energy_profile(self):
        energy_mep = griddata(self.grid, self.V.ravel(), self.mep, method='linear')
        return self.mep, energy_mep

    def plot_V(self, clip_min=None, clip_max=None, levels=None, cmap='RdYlBu', dpi=300):
        """
        Generates a filled contour plot of the energy landscape $V$.

        Args:
            cmap: Colormap for plot.
            levels: Levels to plot contours at (see matplotlib contour/contourf docs for details).
            dpi: DPI.
        """
        fig, ax = plt.subplots(dpi=dpi)

        V = self.V
        if clip_min is not None:
            V = V.clip(min=clip_min)
        if clip_max is not None:
            V = V.clip(max=clip_max)

        cs = ax.contourf(self.X, self.Y, V, levels=levels, cmap=cmap)
        ax.contour(self.X, self.Y, V, levels=levels, colors="black", alpha=0.2)
        cbar = fig.colorbar(cs)
        return fig, ax, cbar

    def plot_mep(self, **plot_V_kwargs):
        """
        Plots the minimum energy path on the energy landscape $V$.

        Args:
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax, cbar = self.plot_V(**plot_V_kwargs)
        ax.scatter(self.mep[0, 0], self.mep[0, 1], color='C0')
        ax.scatter(self.mep[-1, 0], self.mep[-1, 1], color='C0')
        ax.plot(self.mep[:, 0], self.mep[:, 1])
        return fig, ax, cbar

    def plot_mep_energy_profile(self, dpi=300):
        """
        Plots the energy profile along the minimum energy path in $V$.
        """
        energy_mep = griddata(self.grid, self.V.ravel(), self.mep, method='linear')
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(np.linspace(0, 1, len(energy_mep)), energy_mep)
        return fig, ax

    def plot_string_evolution(self, string_cmap=cm.gray, **plot_V_kwargs):
        """
        Plots the evolution of the string on the energy landscape $V$.

        Args:
            string_cmap: Colormap to use for plotting the evolution of the string.
            **plot_V_kwargs: Keyword arguments for plotting the energy landscape V.
        """
        fig, ax, cbar = self.plot_V(**plot_V_kwargs)
        colors = string_cmap(np.linspace(0, 1, len(self.string_traj)))
        for sidx, string in enumerate(self.string_traj):
            ax.plot(string[:, 0], string[:, 1], '--', color=colors[sidx])
        return fig, ax, cbar
