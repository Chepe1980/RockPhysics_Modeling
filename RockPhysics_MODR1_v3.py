import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from io import StringIO
from pyavo.seismodel import tuning_prestack as tp
from pyavo.seismodel import wavelet

# Function for VRH averaging
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

# Function for Fluid Replacement Modeling
def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
    vp1 = vp1 / 1000.
    vs1 = vs1 / 1000.
    mu1 = rho1 * vs1**2.
    k_s1 = rho1 * vp1**2 - (4./3.) * mu1

    # Dry rock bulk modulus
    kdry = (k_s1 * ((phi * k0) / k_f1 + 1 - phi) - k0) / ((phi * k0) / k_f1 + (k_s1 / k0) - 1 - phi)

    # Gassmann's substitution
    k_s2 = kdry + (1 - (kdry / k0))**2 / ((phi / k_f2) + ((1 - phi) / k0) - (kdry / k0**2))
    rho2 = rho1 - phi * rho_f1 + phi * rho_f2
    mu2 = mu1
    vp2 = np.sqrt((k_s2 + (4./3) * mu2) / rho2)
    vs2 = np.sqrt(mu2 / rho2)

    return vp2 * 1000, vs2 * 1000, rho2, k_s2

# Function to generate synthetic gather
def generate_synthetic_gather(vp, vs, rho, scenario_name, thickness=10):
    """Generate synthetic angle gather for given elastic properties"""
    try:
        # Verify input dimensions
        if len(vp) != 3 or len(vs) != 3 or len(rho) != 3:
            raise ValueError("Input arrays must have exactly 3 elements for 3-layer model")
        
        # Create model
        nangles = tp.n_angles(0, 45)  # Generate number of angles (0-45°)
        
        # Calculate reflection coefficients
        rc_zoep = []
        theta1 = []
        
        for angle in range(nangles):
            theta1_samp, rc_1, rc_2 = tp.calc_theta_rc(
                theta1_min=0, 
                theta1_step=1, 
                vp=vp, 
                vs=vs, 
                rho=rho, 
                ang=angle
            )
            theta1.append(theta1_samp)
            rc_zoep.append([rc_1[0, 0], rc_2[0, 0]])
        
        # Generate wavelet
        wlt_time, wlt_amp = wavelet.ricker(
            sample_rate=0.0001, 
            length=0.128, 
            c_freq=30
        )
        
        # Time samples
        t_samp = tp.time_samples(t_min=0, t_max=0.5)
        
        # Generate synthetic seismograms
        syn_zoep = []
        lyr_times = []
        
        for angle in range(nangles):
            # Calculate interface depths and times
            z_int = tp.int_depth(h_int=[500.0], thickness=thickness)
            t_int = tp.calc_times(z_int, vp)
            lyr_times.append(t_int)
            
            # Digitize model and convolve with wavelet
            rc = tp.mod_digitize(rc_zoep[angle], t_int, t_samp)
            s = tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
            syn_zoep.append(s)
        
        # Convert to numpy arrays
        syn_zoep = np.array(syn_zoep)
        rc_zoep = np.array(rc_zoep)
        t = np.array(t_samp)
        lyr_times = np.array(lyr_times)
        
        # Get layer indices
        lyr1_indx, lyr2_indx = tp.layer_index(lyr_times)
        
        # Prepare data for plotting
        top_layer = np.array([syn_zoep[trace, lyr1_indx[trace]] for trace in range(nangles)])
        bottom_layer = np.array([syn_zoep[trace, lyr2_indx[trace]] for trace in range(nangles)])
        
        # Plot the gather
        st.subheader(f"Synthetic Gather: {scenario_name} Scenario")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Corrected syn_angle_gather call
        tp.syn_angle_gather(
            syn_seis=syn_zoep,
            rc_zoep=rc_zoep,
            t=t,
            excursion=2,
            lyr_times=lyr_times,
            thickness=thickness,
            top_layer=top_layer,
            bottom_layer=bottom_layer,
            vp_dig=tp.t_domain(t=t, vp=vp, vs=vs, rho=rho, lyr1_index=lyr1_indx, lyr2_index=lyr2_indx)[0],
            vs_dig=tp.t_domain(t=t, vp=vp, vs=vs, rho=rho, lyr1_index=lyr1_indx, lyr2_index=lyr2_indx)[1],
            rho_dig=tp.t_domain(t=t, vp=vp, vs=vs, rho=rho, lyr1_index=lyr1_indx, lyr2_index=lyr2_indx)[2]
        )
        
        st.pyplot(fig)
        plt.close()
        
        return syn_zoep, rc_zoep
    
    except Exception as e:
        st.error(f"Error generating synthetic gather: {str(e)}")
        st.error(f"Input shapes - VP: {np.shape(vp)}, VS: {np.shape(vs)}, RHO: {np.shape(rho)}")
        return None, None

# Streamlit App
st.title("Fluid Replacement Modeling (FRM) with Synthetic Seismic Generation")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    logs = pd.read_csv(uploaded_file)
    
    # Check required columns
    required_columns = ['VP', 'VS', 'RHO', 'VSH', 'PHI', 'SW', 'DEPTH']
    if not all(col in logs.columns for col in required_columns):
        st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
        st.stop()

    # Sidebar for mineral/fluid properties
    with st.sidebar:
        st.header("Mineral & Fluid Properties")
        st.subheader("Quartz (Sand)")
        rho_qz = st.number_input("Density (g/cc)", value=2.65, key="rho_qz")
        k_qz = st.number_input("Bulk Modulus (GPa)", value=37.0, key="k_qz")
        mu_qz = st.number_input("Shear Modulus (GPa)", value=44.0, key="mu_qz")

        st.subheader("Clay (Shale)")
        rho_sh = st.number_input("Density (g/cc)", value=2.81, key="rho_sh")
        k_sh = st.number_input("Bulk Modulus (GPa)", value=15.0, key="k_sh")
        mu_sh = st.number_input("Shear Modulus (GPa)", value=5.0, key="mu_sh")

        st.subheader("Brine")
        rho_b = st.number_input("Density (g/cc)", value=1.09, key="rho_b")
        k_b = st.number_input("Bulk Modulus (GPa)", value=2.8, key="k_b")

        st.subheader("Oil")
        rho_o = st.number_input("Density (g/cc)", value=0.78, key="rho_o")
        k_o = st.number_input("Bulk Modulus (GPa)", value=0.94, key="k_o")

        st.subheader("Gas")
        rho_g = st.number_input("Density (g/cc)", value=0.25, key="rho_g")
        k_g = st.number_input("Bulk Modulus (GPa)", value=0.06, key="k_g")

        sand_cutoff = st.number_input("Sand/Shale Cutoff (VSH)", value=0.12, key="sand_cutoff")
        selected_depth = st.number_input("Depth for Synthetic Gather (m)", 
                                       min_value=float(logs.DEPTH.min()), 
                                       max_value=float(logs.DEPTH.max()),
                                       value=float(logs.DEPTH.mean()))

    # Process data
    shale = logs.VSH.values
    sand = 1 - shale - logs.PHI.values
    shaleN = shale / (shale + sand)  # Normalized volumes
    sandN = sand / (shale + sand)
    k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])

    # Fluid mixture properties
    water = logs.SW.values
    hc = 1 - logs.SW.values
    _, k_fl, _, _, _, _ = vrh([water, hc], [k_b, k_o], [0, 0])
    rho_fl = water * rho_b + hc * rho_o

    # FRM for brine, oil, gas
    vpb, vsb, rhob, kb = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_b, k_b, k0, logs.PHI)
    vpo, vso, rhoo, ko = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_o, k_o, k0, logs.PHI)
    vpg, vsg, rhog, kg = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_g, k_g, k0, logs.PHI)

    # Classify litho-fluid
    brine_sand = ((logs.VSH <= sand_cutoff) & (logs.SW >= 0.65))
    oil_sand = ((logs.VSH <= sand_cutoff) & (logs.SW < 0.65))
    shale_flag = (logs.VSH > sand_cutoff)

    # Update logs with FRM results
    for suffix, vp, vs, rho in zip(['B', 'O', 'G'], 
                                  [vpb, vpo, vpg], 
                                  [vsb, vso, vsg], 
                                  [rhob, rhoo, rhog]):
        logs[f'VP_FRM{suffix}'] = logs.VP
        logs[f'VS_FRM{suffix}'] = logs.VS
        logs[f'RHO_FRM{suffix}'] = logs.RHO
        logs.loc[brine_sand | oil_sand, f'VP_FRM{suffix}'] = vp[brine_sand | oil_sand]
        logs.loc[brine_sand | oil_sand, f'VS_FRM{suffix}'] = vs[brine_sand | oil_sand]
        logs.loc[brine_sand | oil_sand, f'RHO_FRM{suffix}'] = rho[brine_sand | oil_sand]
        logs[f'IP_FRM{suffix}'] = logs[f'VP_FRM{suffix}'] * logs[f'RHO_FRM{suffix}']
        logs[f'IS_FRM{suffix}'] = logs[f'VS_FRM{suffix}'] * logs[f'RHO_FRM{suffix}']
        logs[f'VPVS_FRM{suffix}'] = logs[f'VP_FRM{suffix}'] / logs[f'VS_FRM{suffix}']

    # LFC (Litho-Fluid Class)
    lfc_mapping = {
        'LFC_B': (brine_sand | oil_sand, 1),
        'LFC_O': (brine_sand | oil_sand, 2),
        'LFC_G': (brine_sand | oil_sand, 3),
        'LFC_SHALE': (shale_flag, 4)
    }
    for col, (condition, value) in lfc_mapping.items():
        logs[col] = np.where(condition, value, 0)

    # Plotting
    ccc = ['#B3B3B3', 'blue', 'green', 'red', '#996633']  # Colors for LFC
    cmap_facies = colors.ListedColormap(ccc, 'indexed')

    # Depth range selection
    depth_min, depth_max = st.slider(
        "Select Depth Range", 
        min_value=int(logs.DEPTH.min()), 
        max_value=int(logs.DEPTH.max()),
        value=(int(logs.DEPTH.min()), int(logs.DEPTH.max()))
    )
    logs_subset = logs[(logs.DEPTH >= depth_min) & (logs.DEPTH <= depth_max)]

    # Figure 1: Log Plots
    st.subheader("Log Plots")
    fig1, ax1 = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
    ax1[0].plot(logs_subset.VSH, logs_subset.DEPTH, '-g', label='Vsh')
    ax1[0].plot(logs_subset.SW, logs_subset.DEPTH, '-b', label='Sw')
    ax1[0].plot(logs_subset.PHI, logs_subset.DEPTH, '-k', label='PHI')
    ax1[1].plot(logs_subset.IP_FRMG, logs_subset.DEPTH, '-r', label='Gas')
    ax1[1].plot(logs_subset.IP_FRMB, logs_subset.DEPTH, '-b', label='Brine')
    ax1[1].plot(logs_subset.IP, logs_subset.DEPTH, '-', color='0.5', label='Original')
    ax1[2].plot(logs_subset.VPVS_FRMG, logs_subset.DEPTH, '-r')
    ax1[2].plot(logs_subset.VPVS_FRMB, logs_subset.DEPTH, '-b')
    ax1[2].plot(logs_subset.VPVS, logs_subset.DEPTH, '-', color='0.5')
    cluster = np.repeat(np.expand_dims(logs_subset['LFC_B'].values, 1), 100, 1)
    im = ax1[3].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=4)

    # Formatting
    for i in ax1[:-1]:
        i.set_ylim(depth_max, depth_min)
        i.grid()
    ax1[0].legend(fontsize='small', loc='lower right')
    ax1[0].set_xlabel("Vcl/PHI/Sw"), ax1[0].set_xlim(-0.1, 1.1)
    ax1[1].set_xlabel("Ip [m/s*g/cc]"), ax1[1].set_xlim(6000, 15000)
    ax1[2].set_xlabel("Vp/Vs"), ax1[2].set_xlim(1.5, 2)
    ax1[3].set_xlabel('LFC')
    ax1[1].legend(fontsize='small')

    # Figure 2: Cross-Plots
    st.subheader("Cross-Plots")
    fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    ax2[0].scatter(logs_subset.IP_FRMB, logs_subset.VPVS_FRMB, 20, logs_subset.LFC_B, 
                  marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
    ax2[1].scatter(logs_subset.IP_FRMO, logs_subset.VPVS_FRMO, 20, logs_subset.LFC_O, 
                  marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
    ax2[2].scatter(logs_subset.IP_FRMG, logs_subset.VPVS_FRMG, 20, logs_subset.LFC_G, 
                  marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
    for a in ax2:
        a.set_xlim(3000, 16000)
        a.set_ylim(1.5, 3)
    ax2[0].set_title("FRM to Brine")
    ax2[1].set_title("FRM to Oil")
    ax2[2].set_title("FRM to Gas")

    # Display plots
    st.pyplot(fig1)
    st.pyplot(fig2)

    # Synthetic Gather Generation
    st.header("Synthetic Seismic Gathers")
    
    # Get data at selected depth
    selected_data = logs_subset[np.abs(logs_subset.DEPTH - selected_depth) < 0.1]
    
    if len(selected_data) > 0:
        # Create base properties (shale)
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
            # Create 3-layer model (shale-target-shale)
            vp_model = [vp_base-100, vp, vp_base+100]  # Small variation
            vs_model = [vs_base-50, vs, vs_base+50]    # Small variation
            rho_model = [rho_base-0.1, rho, rho_base+0.1]  # Small variation
            
            generate_synthetic_gather(
                vp=vp_model,
                vs=vs_model,
                rho=rho_model,
                scenario_name=name
            )
    else:
        st.warning("No data found at selected depth. Please choose another depth.")
