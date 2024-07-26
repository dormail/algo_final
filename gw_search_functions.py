import numpy as np
from scipy import stats
import numba 

"""
This file contains functions that are useful for the exam, and skeletons 
of functions that you will need to complete in the exam.

"""
######################################################################
# Constants
MTSUN_SI = 4.925491025543576e-06 # solar mass in seconds
MCHIRP_RANGE = (1.15 , 1.2) # in solar masses
MASS_RATIO_RANGE = (0.74, 1)
######################################################################
# timestamps corresponding to the strain data
times = np.arange(2048 * 4096 ) / 4096
df = 1 / times[-1]
######################################################################
# Functions given for the exam, that deal with Post Newtonian physics
# of the waveform

def masses_to_pn_coefficients(m1, m2):
    """
    Return the post-Newtonian coefficients for the phase of a waveform 
    given the masses of the binary.
    input: 
    m1 - float or array, primary mass(es), in solar masses
    m2 - float or array, secondary mass(es), in solar masses
    output:
    pn_coefficients - array of post-Newtonian coefficients
    """
    m1, m2, = np.broadcast_arrays(m1, m2 )
    mtot = m1 + m2
    eta = m1 * m2 / mtot**2
    q = m2 / m1
    chis = 0.0
    chia = 0.0
    sx = 0.0
    sy = 0.0
    delta = (m1 - m2) / mtot
    beta = 113/12 * (chis + delta*chia - 76/113*eta*chis)
    
    pn_coefficients = (
        3/128 / eta * mtot**(-5/3),
        3/128 * (55/9 + 3715/756/eta) / mtot,
        3/128 * (4*beta - 16*np.pi) / eta * mtot**(-2/3),
        15/128 * (1+q)**4*(4/3+q)*(sx**2+sy**2)/mtot**4/q**2 * mtot**(-1/3)
        )

    return np.moveaxis(pn_coefficients, 0, -1)  # Last axis is PN order

def pn_coefficients_to_phases(freqs, pn_coefficients):
    """
    Return the phase of a waveform given the post-Newtonian coefficients.
    input:
    freqs - array of frequencies (Hz)
    pn_coefficients - array of post-Newtonian coefficients
    output:
    phase - array of phases
    """
    freqs = np.atleast_1d(freqs)
    i = np.where(freqs > 0)[0][0]
    powers = [-5/3, -3/3, -2/3, -1/3]
    pn_functions = np.zeros((len(freqs), len(powers)))
    pn_functions[i:,] =  - np.power.outer(MTSUN_SI * np.pi * freqs[i:],
                                      powers)
    
    # pn_functions.shape = (len(freqs), len(powers))
    if pn_coefficients.ndim == 1:
        # return a single phase
        phase = pn_functions @ pn_coefficients 
    else:
        # return a phase for each set of coefficients, (n_samples, n_freqs)
        phase = pn_coefficients @ pn_functions.T

    return phase

def mchirp_q_to_m1m2(mchirp, q):
    """Return component masses given chirp mass and mass ratio `q=m2/m1`.
    """
    m1 = mchirp * (1 + q)**.2 / q**.6
    m2 = q * m1
    return m1, m2

def masses_to_phases(m1, m2, freqs):
    """
    Return the phase of a waveform given the masses of the binary. 
    input:
    m1 - array of primary masses (in solar masses)
    m2 - array of secondary masses (in solar masses)
    freqs - array of frequencies (Hz)
    output:
    phase - 2d array of phases (per mass pair, per frequency)
    """
    return pn_coefficients_to_phases(
        freqs, masses_to_pn_coefficients(m1,m2))

def phases_to_linear_free_phases(phases, freqs, weights):
    """
    Return the phases with the linear components removed.
    input:
    phases - array of phases
    freqs - array of frequencies
    weights - array of weights
    output:
    phases_shifted - array of phases with linear components removed
    """
    # find first and second moments of the frequency distribution
    f_bar = np.sum(freqs*weights**2)
    f2_bar = np.sum(freqs**2* weights**2)
    sigma_f = np.sqrt(f2_bar - f_bar**2)
    # define the linear component
    psi_1 = (freqs - f_bar)/sigma_f
    # find the linear component projection on each phase
    c_1 = np.sum(phases * weights**2* psi_1, axis=1)
    # remove linear component from each phase
    phases_shifted = phases - np.outer(c_1, psi_1)
    # fix phase at f_bar to be zero
    phases_shifted -= np.array([np.interp(x=f_bar, xp=freqs, fp=p) 
                                for p in phases_shifted])[..., None]   
    return phases_shifted 

def draw_mass_samples(n_samples):
    """
    Draw n_samples of primary and secondary masses from a fiducial mass
    distribution. Use the chirp-mass and mass ratio a uniform random 
    variables and transoform them to the primary and secondary masses."""
    
    u = stats.qmc.Halton(2).random(n_samples)
    mchirp_samples = stats.uniform(MCHIRP_RANGE[0], 
                                   np.diff(MCHIRP_RANGE)).ppf(u[:,0])
    q_samples = stats.uniform(MASS_RATIO_RANGE[0],
                              np.diff(MASS_RATIO_RANGE)).ppf(u[:,1])
    m1_samples, m2_samples = mchirp_q_to_m1m2(mchirp_samples, q_samples)
    return m1_samples, m2_samples

######################################################################

##
# Functions skeletons to be completed in the exam
##

def correlate(x, y, w=None):
    """
    Return the correlation of two time series.
    Using common time convention, x is the data and y is the template.
    input:
    x - array, frequency domain series.
    y - array, frequency domain series.
    w - array, whitening filter (optional)
    output:
    correlation - array, correlation of x and y timeseires
    """
    # if w is not None, multiply x and y by the whitening filter
    if w is None:
        return 2 * df * np.dot(x, y.conj())
    else:
        return 2 * df * np.sum(x * y.conj() * w**2) # i assumed w = 1/sqrt(S_n(f)) and we already have the factor of S_n(f)/2 in the main file

def coordinates_to_phase(coordinates, phase_basis_vectors, 
                         common_phase_evolution):
    """
    Return the phase of a waveform given coordinates, basis vectors, and
    common phase evolution.
    input:
    coordinates - coordinates of waveform in basis given from SVD. 
    phase_basis_vectors - basis vectors for phase given from SVD (without 
                          inner product weighting)
    common_phase_evolution - common phase evolution of waveforms used to 
                             construct basis vectors
    output:
    phase - phase of waveform
    """
    
    ...

    return phase


def select_points(points, distance_scale):
    """
    Select a subset of the points such that no two points are closer 
    than distance_scale. The function itertates over the points and adds 
    a new point to the subset if it is further than distance_scale from 
    all points already in the subset.

    input:
    points - array of points, shape (n_points, n_dimensions)
    distance_scale - float, minimum distance allowed between points in 
                      the subset
    output:
    subset - array of points, shape (n_subset_points, n_dimensions)
    indices - array of indices of points in subset, 
              shape (n_subset_points,)
    """
    subset = []
    indices = []
    ...
    return np.array(subset), indices

def get_complex_overlap_timeseries(template, data):
    """
    Return a complex times series of the SNR of a template in a data 
    stream.
    input:
    template - array, frequency domain template (whitened)
    data - array, frequency domain data (whitened)
    output:
    overlap_times_series - array of complex overlap time series
    """
    zcos_fft = data * template.conj()
    zsin_fft = 1j * data * template.conj()
    
    zcos = np.fft.irfft(zcos_fft)
    zsin = np.fft.irfft(zsin_fft)

    complex_overlap_timeseries = zcos + 1j * zsin

    return complex_overlap_timeseries

def get_snr2_timeseries(template, data):
    """
    Return a times series of the SNR squred of a template in a data 
    stream. 
    template - array, frequency domain template (whitened)
    data - array, frequency domain data (whitened)
    output:
    snr2_times_series - array of (real) SNR^2 time series
    """
    z = get_complex_overlap_timeseries(template, data)
    snr2_timeseries = np.abs(z) ** 2

    return snr2_timeseries


def svd_of_phases(phases, weights=None, n_modes=None):
    """
    Return the singular value decomposition of a set of phases.
    input:
    phases - array, phases of waveforms
    weights - array, weights for inner product
    n_modes - int, number of modes to keep
    output:
    u - array, left singular vectors
    s - array, singular values
    v - array, right singular vectors
    """
    # impose normalization on the weights
    ...
    # perpare the phases for SVD. Remove the common phase evolution and
    # project-out the constant phase
    ...

    return u, s, v, common_phase_evolution

@numba.njit()
def max_argmax_over_n_samples(x, n):
    """
    Find the maximum and argumax of x each segments of length n.
    input:
    x - array, input array
    n - int, length of segments
    output:
    maxs - array, maximum of each segment
    argmaxs - array, argmax of each segment
    """
    ...
    return maxs, argmaxs