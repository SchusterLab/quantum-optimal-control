"""
plotutil.py - A module for plotting utilities.
"""

from itertools import product
import ntpath
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la

# define constants
COLOR_PALETTE = ["blue", "red", "green", "pink", "purple", "orange", "teal",
                 "grey", "black", "cyan", "magenta", "brown", "azure", "beige",
                 "coral", "crimson"]

def plot_uks(file_path, save_path=None,
             amplitude_unit="GHz", time_unit="ns",
             dpi=1000, marker_style="o"):
    """
    Get the final uks from a grape H5 file and plot them as a png.
    Args:
    file_path :: str - the full path to the H5 file
    save_path :: str - the full path to save the png file to
    amplitude_unit :: str - the unit to display for the pulse amplitude
    time_unit :: str - the unit to display for the pulse duration
    dpi :: int - the quality of the image
    marker_style :: str - the style to plot as
    Returns: nothing
    """
    # Open file and extract data.
    file_name = os.path.splitext(ntpath.basename(file_path))[0]
    f = h5py.File(file_path, "r")
    # Get the last set of uks.
    uks = np.array(f["uks"][-1])
    uks_fft = list()
    step_count = len(uks[0])
    h_names = list(f["Hnames"])
    pulse_time = f["total_time"][()]
    time_per_step = np.divide(pulse_time, step_count)
    step_per_time = np.divide(step_count, pulse_time)

    # If the user did not specify a save path,
    # save the file to the current directory with
    # the data file's prefix.
    if save_path is None:
        save_path = "{}.png".format(file_name)

    # Get population data via density matrices.
    initial_vectors = f["initial_vectors_c"]
    initial_vector_count = len(initial_vectors)
    # transform row vectors to column vectors
    vectors = [np.vstack(vec) for vec in initial_vectors]
    drift_hamiltonian = f["H0"]
    control_hamiltonians = f["Hops"]
    state_population_data = list()
    for _ in vectors:
        state_population_data.append(list())
    hilbert_space_dimension = len(drift_hamiltonian)
    for i in range(step_count):
        h = drift_hamiltonian
        for j, hc in enumerate(control_hamiltonians):
            h += uks[j][i] * hc
        u = la.expm(1j * h * time_per_step)
        vectors = np.matmul(u, vectors)
        probability_density_matrices = [np.abs(np.outer(vec, vec)) for vec in vectors]
        for j, probability_density_matrix in enumerate(probability_density_matrices):
            state_occupations = list()
            for state_index in range(hilbert_space_dimension):
                state_occupations.append(probability_density_matrix[state_index][state_index])
            state_population_data[j].append(state_occupations)
        #ENDFOR
    #ENDFOR
    # reshape the pop data so it is index by vector -> state -> time step
    # not vector -> time step -> state
    population_data = list()
    for i in range(initial_vector_count):
        vec_pop_data = list()
        for j in range(hilbert_space_dimension):
            vec_pop_data.append(list())
        #ENDFOR
        population_data.append(vec_pop_data)
    #ENDFOR

    for i in range(initial_vector_count):
        for j in range(hilbert_space_dimension):
            for k in range(step_count):
                val = state_population_data[i][k][j]
                population_data[i][j].append(val)
            #ENDFOR
        #ENDFOR
    #ENDFOR

    # Create labels and extra content.
    patches = list()
    labels = h_names
    for i, pulse in enumerate(uks):
        label = labels[i]
        color = COLOR_PALETTE[i]
        patches.append(mpatches.Patch(label=label, color=color))
        uks_fft.append(np.power(np.abs(np.fft.fft(pulse)), 2))
    #ENDFOR

    # Plot the data.
    plt.figure()
    plt.suptitle(file_name)
    plt.figlegend(handles=patches, labels=labels, loc="upper right",
                  framealpha=0.5)
    plt.subplots_adjust(hspace=0.8)
    subplot_count = 2 + initial_vector_count

    # pulses
    plt.subplot(subplot_count, 1, 1)
    time_axis = time_per_step * np.arange(step_count)
    for i, pulse in enumerate(uks):
        color = COLOR_PALETTE[i]
        plt.plot(time_axis, pulse, marker_style,
                 color=color, ms=2, alpha=0.9)
    #ENDFOR
    plt.xlabel("Time ({})".format(time_unit))
    plt.ylabel("Amplitude ({})".format(amplitude_unit))

    # fft
    plt.subplot(subplot_count, 1, 2)
    freq_axis = np.arange(step_count) * np.divide(step_per_time, step_count)
    for i, pulse_fft in enumerate(uks_fft):
        color = COLOR_PALETTE[i]
        plt.plot(freq_axis,
                 pulse_fft, marker_style, color=color,
                 ms=2,alpha=0.9)
    #ENDFOR
    plt.xlabel("Frequency ({})".format(amplitude_unit))
    plt.ylabel("FFT")

    # state population
    for i in range(initial_vector_count):
        plt.subplot(subplot_count, 1, 3 + i)
        plt.xlabel("Time ({})".format(time_unit))
        plt.ylabel("Population")
        for j in range(hilbert_space_dimension):
            pop_data = population_data[i][j]
            color = COLOR_PALETTE[j]
            if i == 0:
                print("hs: {}, c: {}".format(j, color))
            plt.plot(time_axis, pop_data, marker_style, color=color, ms=2, alpha=0.5)
        #ENDFOR
    #ENDFOR
    
    plt.savefig(save_path, dpi=dpi)
    

def _tests():
    """
    Run tests on the module.
    """
    pass


if __name__ == "__main__":
   _tests()
