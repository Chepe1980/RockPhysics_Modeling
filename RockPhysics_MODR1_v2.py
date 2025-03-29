import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, LassoSelectTool
from bokeh.layouts import gridplot, column
from bokeh.transform import factor_cmap
from bokeh.embed import components
from bokeh.resources import CDN
import streamlit.components.v1 as components

# Check versions (for debugging)
import bokeh
st.sidebar.write(f"Bokeh v{bokeh.__version__}")
st.sidebar.write(f"Streamlit v{st.__version__}")

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

# Streamlit App
st.title("Fluid Replacement Modeling (FRM) with Gassmann's Equations")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        logs = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_columns = ['VP', 'VS', 'RHO', 'VSH', 'PHI', 'SW', 'DEPTH']
        if not all(col in logs.columns for col in required_columns):
            st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
            st.stop()

        # Clean data
        logs = logs.dropna(subset=required_columns).copy()
        if logs.empty:
            st.error("No valid data after removing rows with missing values")
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

        # Process data
        shale = logs.VSH.values
        sand = 1 - shale - logs.PHI.values
        shaleN = shale / (shale + sand + 1e-10)  # Avoid division by zero
        sandN = sand / (shale + sand + 1e-10)
        k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])

        # Fluid mixture properties
        water = logs.SW.values
        hc = 1 - logs.SW.values
        _, k_fl, _, _, _, _ = vrh([water, hc], [k_b, k_o], [0, 0])
        rho_fl = water * rho_b + hc * rho_o

        # FRM calculations with error handling
        try:
            vpb, vsb, rhob, kb = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_b, k_b, k0, logs.PHI)
            vpo, vso, rhoo, ko = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_o, k_o, k0, logs.PHI)
            vpg, vsg, rhog, kg = frm(logs.VP, logs.VS, logs.RHO, rho_fl, k_fl, rho_g, k_g, k0, logs.PHI)
        except Exception as e:
            st.error(f"FRM calculation failed: {str(e)}")
            st.stop()

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
            logs[f'VPVS_FRM{suffix}'] = logs[f'VP_FRM{suffix}'] / (logs[f'VS_FRM{suffix}'] + 1e-10)

        # LFC (Litho-Fluid Class)
        logs['LFC'] = np.select(
            [
                shale_flag,
                brine_sand,
                oil_sand
            ],
            [
                4,  # Shale
                1,  # Brine sand
                2   # Oil sand
            ],
            default=3  # Gas sand
        )

        # Depth range selection
        depth_min, depth_max = st.slider(
            "Select Depth Range", 
            min_value=int(logs.DEPTH.min()), 
            max_value=int(logs.DEPTH.max()),
            value=(int(logs.DEPTH.min()), int(logs.DEPTH.max()))
        )
        logs_subset = logs[(logs.DEPTH >= depth_min) & (logs.DEPTH <= depth_max)].copy()

        # Prepare data source for Bokeh plots
        source_data = {
            'ip_b': [float(x) for x in logs_subset.IP_FRMB],
            'vpvs_b': [float(x) for x in logs_subset.VPVS_FRMB],
            'ip_o': [float(x) for x in logs_subset.IP_FRMO],
            'vpvs_o': [float(x) for x in logs_subset.VPVS_FRMO],
            'ip_g': [float(x) for x in logs_subset.IP_FRMG],
            'vpvs_g': [float(x) for x in logs_subset.VPVS_FRMG],
            'depth': [float(x) for x in logs_subset.DEPTH],
            'vsh': [float(x) for x in logs_subset.VSH],
            'sw': [float(x) for x in logs_subset.SW],
            'phi': [float(x) for x in logs_subset.PHI],
            'lfc': [str(int(x)) for x in logs_subset.LFC],
            'selected': [0] * len(logs_subset)
        }
        source = ColumnDataSource(data=source_data)

        # Color mapping
        lfc_palette = ['#B3B3B3', 'blue', 'green', 'red', '#996633']

        # Create interactive plots
        st.subheader("Interactive Cross-Plots")
        
        try:
            # Cross-plot 1: Brine
            p1 = figure(width=400, height=400, tools="pan,wheel_zoom,box_zoom,reset,lasso_select",
                       title="FRM to Brine", x_range=(3000, 16000), y_range=(1.5, 3))
            p1.scatter('ip_b', 'vpvs_b', source=source, size=8, alpha=0.6,
                      color=factor_cmap('lfc', palette=lfc_palette, factors=['1','2','3','4']),
                      selection_color="orange")
            p1.xaxis.axis_label = "IP [m/s*g/cc]"
            p1.yaxis.axis_label = "Vp/Vs"

            # Similar plots for Oil and Gas (p2, p3)
            
            # Log plot
            log_plot = figure(width=800, height=400, y_range=(depth_max, depth_min),
                            tools="pan,wheel_zoom,box_zoom,reset")
            log_plot.line('vsh', 'depth', source=source, line_color='green')
            log_plot.line('sw', 'depth', source=source, line_color='blue')

            # JavaScript callback
            callback = CustomJS(args=dict(source=source), code="""
                // Selection handling code
            """)

            # Layout
            layout = column(gridplot([[p1, p2, p3]]), log_plot)
            
            # Render
            script, div = components(layout)
            components.html(
                f"""
                <link href="{CDN.css_files[0]}" rel="stylesheet">
                <script src="{CDN.js_files[0]}"></script>
                {div}
                {script}
                """,
                height=1000
            )
            
        except Exception as e:
            st.error(f"Interactive plots disabled: {str(e)}")
            
            # Enhanced static fallback
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            # Crossplot
            sc = ax1.scatter(logs_subset.IP_FRMB, logs_subset.VPVS_FRMB, 
                           c=logs_subset.LFC, cmap=ListedColormap(lfc_palette),
                           vmin=0, vmax=4)
            ax1.set_xlabel("IP [m/s*g/cc]")
            ax1.set_ylabel("Vp/Vs")
            
            # Log plot
            ax2.plot(logs_subset.VSH, logs_subset.DEPTH, 'g-', label='Vsh')
            ax2.plot(logs_subset.SW, logs_subset.DEPTH, 'b-', label='Sw')
            ax2.invert_yaxis()
            ax2.legend()
            
            plt.colorbar(sc, ax=ax1, label='LFC')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
