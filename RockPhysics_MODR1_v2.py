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
import streamlit.components.v1 as components

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

    # Process data
    shale = logs.VSH.values
    sand = 1 - shale - logs.PHI.values
    shaleN = shale / (shale + sand)
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

    # Depth range selection
    depth_min, depth_max = st.slider(
        "Select Depth Range", 
        min_value=int(logs.DEPTH.min()), 
        max_value=int(logs.DEPTH.max()),
        value=(int(logs.DEPTH.min()), int(logs.DEPTH.max()))
    )
    logs_subset = logs[(logs.DEPTH >= depth_min) & (logs.DEPTH <= depth_max)]

    # Prepare data source for Bokeh plots
    source = ColumnDataSource(data=dict(
        ip_b=logs_subset.IP_FRMB,
        vpvs_b=logs_subset.VPVS_FRMB,
        ip_o=logs_subset.IP_FRMO,
        vpvs_o=logs_subset.VPVS_FRMO,
        ip_g=logs_subset.IP_FRMG,
        vpvs_g=logs_subset.VPVS_FRMG,
        depth=logs_subset.DEPTH,
        vsh=logs_subset.VSH,
        sw=logs_subset.SW,
        phi=logs_subset.PHI,
        lfc_b=logs_subset.LFC_B.astype(str),
        lfc_o=logs_subset.LFC_O.astype(str),
        lfc_g=logs_subset.LFC_G.astype(str),
        selected=np.zeros(len(logs_subset))
    ))

    # Color mapping
    lfc_palette = ['#B3B3B3', 'blue', 'green', 'red', '#996633']

    # Create all interactive plots
    st.subheader("Interactive Cross-Plots with Lasso Selection")
    st.markdown("""
        **Instructions**: 
        1. Click the lasso tool ( ) in the toolbar
        2. Draw a shape around points to select them
        3. Selected points will highlight in orange across all plots
    """)

    # Common tools for all plots
    TOOLS = "pan,wheel_zoom,box_zoom,reset,lasso_select"
    plot_size = 400

    # Cross-plot 1: Brine
    p1 = figure(tools=TOOLS, width=plot_size, height=plot_size,
               title="FRM to Brine", x_range=(3000, 16000), y_range=(1.5, 3))
    p1.scatter('ip_b', 'vpvs_b', source=source, size=8, alpha=0.6,
              color=factor_cmap('lfc_b', palette=lfc_palette, factors=['1','2','3','4']),
              selection_color="orange", nonselection_alpha=0.1)
    p1.xaxis.axis_label = "IP [m/s*g/cc]"
    p1.yaxis.axis_label = "Vp/Vs"

    # Cross-plot 2: Oil
    p2 = figure(tools=TOOLS, width=plot_size, height=plot_size,
               title="FRM to Oil", x_range=p1.x_range, y_range=p1.y_range)
    p2.scatter('ip_o', 'vpvs_o', source=source, size=8, alpha=0.6,
              color=factor_cmap('lfc_o', palette=lfc_palette, factors=['1','2','3','4']),
              selection_color="orange", nonselection_alpha=0.1)
    p2.xaxis.axis_label = "IP [m/s*g/cc]"
    p2.yaxis.axis_label = "Vp/Vs"

    # Cross-plot 3: Gas
    p3 = figure(tools=TOOLS, width=plot_size, height=plot_size,
               title="FRM to Gas", x_range=p1.x_range, y_range=p1.y_range)
    p3.scatter('ip_g', 'vpvs_g', source=source, size=8, alpha=0.6,
              color=factor_cmap('lfc_g', palette=lfc_palette, factors=['1','2','3','4']),
              selection_color="orange", nonselection_alpha=0.1)
    p3.xaxis.axis_label = "IP [m/s*g/cc]"
    p3.yaxis.axis_label = "Vp/Vs"

    # Log plot
    log_plot = figure(width=800, height=400, title="Selected Logs",
                     y_range=(depth_max, depth_min), x_range=(0, 1),
                     tools="pan,wheel_zoom,box_zoom,reset")
    log_plot.line('vsh', 'depth', source=source, line_color='green', legend_label='Vsh')
    log_plot.line('sw', 'depth', source=source, line_color='blue', legend_label='Sw')
    log_plot.line('phi', 'depth', source=source, line_color='black', legend_label='PHI')
    log_plot.xaxis.axis_label = "Normalized Values"
    log_plot.yaxis.axis_label = "Depth"

    # Add selected points to log plot
    selected_renderer = log_plot.scatter(x=0.5, y='depth', source=source, size=10,
                                       color='orange', alpha=0,
                                       selection_color="orange",
                                       selection_alpha=0.8)

    # JavaScript callback for interactivity
    callback = CustomJS(args=dict(source=source, selected_renderer=selected_renderer), code="""
        const selected_indices = source.selected.indices;
        const data = source.data;
        
        // Update selection array
        data['selected'] = Array(data['depth'].length).fill(0);
        for (let i = 0; i < selected_indices.length; i++) {
            data['selected'][selected_indices[i]] = 1;
        }
        
        // Make selected points visible in log plot
        selected_renderer.glyph.alpha = {value: 0.8};
        
        source.change.emit();
    """)

    # Add callback to all cross-plots
    for p in [p1, p2, p3]:
        p.js_on_event('selectiongeometry', callback)
        p.select(LassoSelectTool).select_every_mousemove = False

    # Create layout
    cross_plots = gridplot([[p1, p2, p3]], toolbar_location='right')
    full_layout = column(cross_plots, log_plot)

    # Generate components
    script, div = components(full_layout)
    
    # Display using HTML with proper Bokeh resources
    components.html(
        f"""
        <link href="https://cdn.bokeh.org/bokeh/release/bokeh-3.3.4.min.css" rel="stylesheet">
        <link href="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.3.4.min.css" rel="stylesheet">
        <link href="https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.3.4.min.css" rel="stylesheet">
        
        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.3.4.min.js"></script>
        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.3.4.min.js"></script>
        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.3.4.min.js"></script>
        
        <style>
            .bk-tool-icon-lasso-select {{
                visibility: visible !important;
            }}
            .bk-toolbar-button {{
                border: 1px solid #ccc;
                border-radius: 4px;
                margin: 2px;
            }}
        </style>
        
        {div}
        {script}
        """,
        height=900
    )

    # Static plots for reference
    st.subheader("Static Plots for Reference")
    
    # Define color map for facies
    ccc = ['#B3B3B3', 'blue', 'green', 'red', '#996633']
    cmap_facies = ListedColormap(ccc, 'indexed')
    
    # Figure 1: Log Plots
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

    st.pyplot(fig1)
