import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from io import StringIO
from pyavo.seismodel import tuning_prestack as tp
from pyavo.seismodel import wavelet

# Custom synthetic gather visualization
def plot_angle_gather(seismic_data, times, angles, vp):
    """
    Custom function to plot angle gathers
    Args:
        seismic_data: 2D array of seismic traces [nangles x nsamples]
        times: 1D array of time values
        angles: 1D array of angles
        vp: P-wave velocities for depth conversion
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize data for display
    norm_data = seismic_data / np.max(np.abs(seismic_data))
    
    # Create offset approximation from angles
    depth = 500  # Approximate depth in meters
    offsets = depth * np.tan(np.radians(angles))
    
    # Plot each trace
    for i, angle in enumerate(angles):
        trace = norm_data[i] + offsets[i]/1000  # Scale offsets for display
        ax.plot(trace, times, 'k', linewidth=0.5)
        ax.text(trace[-1], times[-1], f'{angle}°', 
                ha='left', va='top', fontsize=8)
    
    # Format plot
    ax.set_xlabel('Offset (km)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Synthetic Angle Gather')
    ax.set_ylim(times[-1], times[0])
    ax.grid(True)
    
    return fig

# VRH averaging function
def vrh(volumes, k, mu):
    f = np.array(volumes).T
    k = np.resize(np.array(k), np.shape(f))
    mu = np.resize(np.array(mu), np.shape(f))

    k_u = np.sum(f * k, axis=1)
    k_l = 1. / np.sum(f / k, axis=1)
    mu_u = np.sum(f * mu, axis=1)
    mu_l = 1. / np.sum(f / mu, axis=1)
    k0 = (k_u + k_l) / 2.
    mu0 = (mu_u + mu_l) / 2.
    return k_u, k_l, mu_u, mu_l, k0, mu0

# Fluid Replacement Modeling
def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
    vp1 = vp1 / 1000.
    vs1 = vs1 / 1000.
    mu1 = rho1 * vs1**2.
    k_s1 = rho1 * vp1**2 - (4./3.) * mu1

    kdry = (k_s1 * ((phi * k0) / k_f1 + 1 - phi) - k0) / ((phi * k0) / k_f1 + (k_s1 / k0) - 1 - phi)

    k_s2 = kdry + (1 - (kdry / k0))**2 / ((phi / k_f2) + ((1 - phi) / k0) - (kdry / k0**2))
    rho2 = rho1 - phi * rho_f1 + phi * rho_f2
    mu2 = mu1
    vp2 = np.sqrt((k_s2 + (4./3) * mu2) / rho2)
    vs2 = np.sqrt(mu2 / rho2)

    return vp2 * 1000, vs2 * 1000, rho2, k_s2

# Generate synthetic gather with custom plotting
def generate_synthetic_gather(vp, vs, rho, scenario_name):
    try:
        # Create model
        nangles = 10  # Number of angles from 0-45°
        angles = np.linspace(0, 45, nangles)
        
        # Calculate reflection coefficients
        rc_zoep = []
        for angle in angles:
            _, rc_1, rc_2 = tp.calc_theta_rc(
                theta1_min=0,
                theta1_step=1,
                vp=vp,
                vs=vs,
                rho=rho,
                ang=angle
            )
            rc_zoep.append([rc_1[0,0], rc_2[0,0]])
        
        # Generate wavelet
        _, wlt_amp = wavelet.ricker(sample_rate=0.0001, length=0.128, c_freq=30)
        
        # Time samples
        t_samp = tp.time_samples(t_min=0, t_max=0.5)
        
        # Generate synthetic seismograms
        syn_zoep = []
        z_int = tp.int_depth(h_int=[500.0], thickness=10)
        t_int = tp.calc_times(z_int, vp)
        
        for angle in range(nangles):
            rc = tp.mod_digitize(rc_zoep[angle], t_int, t_samp)
            s = tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
            syn_zoep.append(s)
        
        syn_zoep = np.array(syn_zoep)
        
        # Plot with our custom function
        st.subheader(f"Synthetic Gather: {scenario_name} Scenario")
        fig = plot_angle_gather(syn_zoep, t_samp, angles, vp)
        st.pyplot(fig)
        plt.close()
        
        return syn_zoep
    
    except Exception as e:
        st.error(f"Error generating synthetic gather: {str(e)}")
        return None

# Streamlit App
st.title("Fluid Replacement Modeling with Synthetic Seismic")

# File upload and processing
uploaded_file = st.file_uploader("Upload well logs CSV", type=["csv"])
if uploaded_file is not None:
    logs = pd.read_csv(uploaded_file)
    
    # Check required columns
    required_cols = ['VP', 'VS', 'RHO', 'VSH', 'PHI', 'SW', 'DEPTH']
    if not all(col in logs.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
        st.stop()

    # Sidebar controls
    with st.sidebar:
        st.header("Mineral & Fluid Properties")
        # [Mineral and fluid property inputs...]
        
        selected_depth = st.slider("Analysis depth (m)", 
                                 float(logs.DEPTH.min()),
                                 float(logs.DEPTH.max()),
                                 float(logs.DEPTH.mean()))

    # Process data
    # [VRH averaging, FRM calculations...]
    
    # Generate and display synthetic gathers
    st.header("Synthetic Seismic Gathers")
    selected_data = logs.iloc[(logs.DEPTH - selected_depth).abs().argsort()[:1]]
    
    if len(selected_data) > 0:
        # Create 3-layer model
        vp_base = selected_data.VP.values[0]
        vs_base = selected_data.VS.values[0]
        rho_base = selected_data.RHO.values[0]
        
        # Generate for each scenario
        scenarios = {
            'Brine': (selected_data['VP_FRMB'].values[0],
                     selected_data['VS_FRMB'].values[0],
                     selected_data['RHO_FRMB'].values[0]),
            'Oil': (selected_data['VP_FRMO'].values[0],
                   selected_data['VS_FRMO'].values[0],
                   selected_data['RHO_FRMO'].values[0]),
            'Gas': (selected_data['VP_FRMG'].values[0],
                   selected_data['VS_FRMG'].values[0],
                   selected_data['RHO_FRMG'].values[0])
        }
        
        for name, (vp, vs, rho) in scenarios.items():
            vp_model = [vp_base-100, vp, vp_base+100]
            vs_model = [vs_base-50, vs, vs_base+50]
            rho_model = [rho_base-0.1, rho, rho_base+0.1]
            
            generate_synthetic_gather(vp_model, vs_model, rho_model, name)
