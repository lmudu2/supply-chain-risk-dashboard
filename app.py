import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import random

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="SupplyGuard 360", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

# --- CSS: HIGH VISIBILITY ---
st.markdown("""
    <style>
        /* 1. Force Page Background */
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }

        /* 2. Force Input Box Background */
        .stMultiSelect div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border-color: #d0d0d0 !important;
            color: #000000 !important;
        }

        /* 3. Force Dropdown Menu Background */
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {
            background-color: #ffffff !important;
            border-color: #d0d0d0 !important;
        }

        /* 4. Force List Items (The Options) to be White/Black */
        li[role="option"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        /* 5. FIX THE BLACK HOVER / FOCUS STATE */
        /* This targets the item when your mouse is over it OR when you arrow-key down to it */
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background-color: #f0f2f6 !important; /* Light Grey Highlight */
            color: #000000 !important;
        }
        .stDataFrame table thead th {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        /* 6. Fix the "Selected Tags" inside the box */
        div[data-baseweb="tag"] {
            background-color: #e0e2e6 !important;
        }
        div[data-baseweb="tag"] span {
            color: #000000 !important;
        }

        /* 7. General Text Overrides */
        h1, h2, h3, h4, p, span, div, label {
            color: #000000 !important;
        }

        /* Hide Sidebar */
        section[data-testid="stSidebar"] { display: none; }
        .block-container { padding: 1rem 2rem; }
        div[data-baseweb="select"] > div {
            background-color: white !important;
            color: black !important;
            border-color: #d0d0d0 !important;
        }
        
        /* 2. Number Input (Order Value) */
        div[data-baseweb="input"] > div {
            background-color: white !important;
            color: black !important;
            border-color: #d0d0d0 !important;
        }
        input[type="number"] {
            color: black !important;
        }
        
        /* 3. Sliders (Risk Index, Capacity) */
        div[data-baseweb="slider"] {
            color: black !important;
        }
        
        /* 4. Submit Button (Run Prediction) */
        button[data-testid="stFormSubmitButton"] {
            background-color: #0066cc !important; /* Blue Button */
            color: white !important;              /* White Text */
            border: none !important;
        }
        button[data-testid="stFormSubmitButton"]:hover {
            background-color: #004c99 !important; /* Darker Blue on Hover */
            color: white !important;
        }
        
        /* 5. Force Labels to be Black */
        label {
            color: black !important;
        }
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-color: #000000 !important; /* Force Black Border */
        }
        
        /* Force the text inside inputs to be black */
        input[type="number"], div[data-baseweb="select"] span {
            color: #000000 !important;
        }
        
        /* --- 2. FORCE SUBMIT BUTTON TO WHITE/BLACK --- */
        /* Target the actual button element inside the Streamlit form button container */
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #ffffff !important; /* White Background */
            color: #000000 !important;            /* Black Text */
            border: 2px solid #000000 !important; /* Solid Black Border */
            font-weight: bold !important;
        }

        /* Change color on Hover so you know it's clickable */
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #f0f0f0 !important; /* Light Grey on Hover */
            color: #000000 !important;
            border: 2px solid #000000 !important;
        }

        /* --- 3. FORCE LABELS (e.g. "Order Value ($)") TO BLACK --- */
        label, .stMarkdown p {
            color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA GENERATOR (NOW WITH PRODUCT RISK LOGIC) ---
@st.cache_resource
def get_data():
    rows = []
    vendors = [f'Global_Supplier_{str(i).zfill(3)}' for i in range(1, 35)]
    modes = ['Air', 'Land', 'Sea']
    
    # PRODUCT COMPLEXITY MAP (Risk Adder)
    # Higher number = Higher inherent risk of delay/damage
    product_risk_map = {
        'Microchips': 0.20,          # Fragile + High Value
        'Medical Devices': 0.25,     # Regulated + Fragile
        'Lithium Batteries': 0.30,   # Hazmat + Dangerous
        'Consumer Electronics': 0.15,# High Value
        'Auto Parts': 0.10,          # Heavy + Custom
        'Chemicals': 0.20,           # Hazmat
        'Steel Beams': 0.05,         # Standard
        'Textiles': 0.00             # Low Risk
    }
    products = list(product_risk_map.keys())

    defect_map = {
        'Microchips': ['Calibration Error', 'Temperature Excursion', 'Software Bug', 'ESD Damage'],
        'Medical Devices': ['Sterilization Failure', 'Software Bug', 'Calibration Error', 'Packaging Breach'],
        'Lithium Batteries': ['Leakage', 'Overheating Risk', 'Voltage Instability', 'Labeling Error'],
        'Consumer Electronics': ['Software Bug', 'Battery Drain', 'Cosmetic Scratch', 'Dead Pixel'],
        'Auto Parts': ['Dimension Mismatch', 'Rust/Corrosion', 'Material Fatigue', 'Welding Defect'],
        'Chemicals': ['Impurity Found', 'Container Leak', 'Concentration Error', 'Labeling Error'],
        'Steel Beams': ['Rust/Corrosion', 'Dimension Mismatch', 'Tensile Strength Fail', 'Cracking'],
        'Textiles': ['Color Mismatch', 'Stitching Defect', 'Fabric Tear', 'Pattern Error'] 
    }
    
    # Mappings
    region_map = {
        'Region_US_East': ['USA', 'Canada', 'Mexico', 'Panama', 'Costa Rica'],
        'Region_EU_Central': ['Germany', 'France', 'Poland', 'Italy', 'Netherlands', 'Spain', 'Sweden'],
        'Region_Asia_Pacific': ['China', 'Vietnam', 'Thailand', 'India', 'Japan', 'South Korea', 'Malaysia'],
        'Region_LatAm_South': ['Brazil', 'Argentina', 'Chile', 'Peru', 'Colombia'],
        'Region_Africa_West': ['Nigeria', 'Ghana', 'Ivory Coast', 'Senegal', 'Cameroon']
    }
    regions = list(region_map.keys())
    
    # Root Causes
    geo_reasons = ['Trade Tariffs', 'Port Strike', 'Civil Unrest', 'Border Closure', 'Sanctions', 'Political Instability']
    climate_zones = ['Typhoon Belt', 'Flood Plain', 'Wildfire Zone', 'Earthquake Fault', 'Monsoon Region']
    ops_reasons = ['Raw Material Shortage', 'Port Congestion', 'Machinery Breakdown', 'Customs Hold', 'Labor Strike']
    qual_reasons = ['Calibration Error', 'Packaging Damage', 'Impure Raw Material', 'Software Bug', 'Temperature Excursion']
    rel_reasons = ['Slow Response', 'Contract Dispute', 'Management Change', 'Pricing Conflict', 'Lack of Transparency']
    fin_reasons = ['High Debt Ratio', 'Liquidity Crisis', 'Pending Lawsuit', 'Merger Uncertainty', 'Credit Downgrade']

    for i in range(10000):
        vendor = random.choice(vendors)
        region = random.choice(regions)
        country = random.choice(region_map[region])
        mode = random.choice(modes)
        product = random.choice(products)
        
        # Risk Logic
        base_risk = {'Region_US_East': 10, 'Region_EU_Central': 20, 'Region_Asia_Pacific': 40, 'Region_LatAm_South': 70, 'Region_Africa_West': 90}[region]
        country_variance = random.randint(-15, 15)
        risk_idx = int(base_risk + country_variance + np.random.normal(0, 5))
        risk_idx = max(0, min(100, risk_idx))
        
        cap_util = int(np.random.normal(70, 15))
        cap_util = max(0, min(100, cap_util))
        val = random.randint(5000, 500000)
        
        # --- CUMULATIVE PROBABILITY (Updated with Product Risk) ---
        prob_late = 0.05 
        if risk_idx > 60: prob_late += 0.30   # High Country Risk
        if cap_util > 85: prob_late += 0.25   # Factory Overload
        if val > 200000: prob_late += 0.05    # High Value
        if mode == 'Sea': prob_late += 0.10   # Slow Mode
        
        # ADD PRODUCT RISK
        prob_late += product_risk_map[product]
        
        # Outcome
        is_late = 1 if random.random() < prob_late else 0
        delay = random.randint(3, 20) if is_late else 0
        
        # Root Causes
        geo_reason = random.choice(geo_reasons) if risk_idx > 60 else "None"
        climate_zone = random.choice(climate_zones) if risk_idx > 70 else "Safe Zone"
        climate_risk = 'High' if risk_idx > 70 else 'Low'
        ops_reason = random.choice(ops_reasons) if is_late else "On Time"
        
        credit = np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'C'], p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
        fin_reason = random.choice(fin_reasons) if credit in ['C', 'BB'] else "Stable"
        


        defect_rate = np.random.uniform(0, 0.05)
        defect_flag = 1 if defect_rate > 0.03 else 0 
        qual_reason = random.choice(qual_reasons) if defect_flag else "Perfect"
        compliance_status = 'Fail' if random.random() < 0.02 else 'Pass'
        if defect_flag:
            # THIS ENSURES TEXTILES GET TEXTILE DEFECTS
            qual_reason = random.choice(defect_map[product])
        else:
            qual_reason = "Perfect"
        
        collab_score = round(random.uniform(1, 10), 1)
        rel_reason = random.choice(rel_reasons) if collab_score < 4.0 else "Good Standing"

        rows.append({
            'Order_ID': f"PO-{20240000+i}",
            'Product': product,
            'Vendor': vendor, 'Region': region, 'Country': country, 'Shipment_Mode': mode,
            'Order_Value_USD': val,
            
            # Geo
            'Country_Risk_Index': risk_idx,
            'Risk_Reason_Geo': geo_reason, 
            'Natural_Disaster_Risk': climate_risk,
            'Climate_Zone_Detail': climate_zone,
            
            # Fin
            'Supplier_Credit_Rating': credit,
            'Financial_Risk_Reason': fin_reason,
            'DPO_Impact_Days': np.random.randint(30, 90),
            'Cost_Competitiveness': int(np.random.normal(100, 5)),
            
            # Ops
            'Is_Late': is_late, 'Delay_Days': delay,
            'Delay_Root_Cause': ops_reason,
            'Order_Accuracy_Rate': int(min(100, np.random.normal(98, 2))),
            'Capacity_Utilization': cap_util,
            
            # Qual
            'Defect_Rate': defect_rate,
            'Defect_Flag': defect_flag,
            'Defect_Root_Cause': qual_reason,
            'Audit_Status': np.random.choice(['Certified', 'Pending', 'Failed'], p=[0.7, 0.2, 0.1]),
            'Regulatory_Compliance': compliance_status,
            'ESG_Score': int(np.random.normal(75, 15)),
            
            # Rel
            'Responsiveness_Score': int(np.random.normal(85, 10)),
            'Innovation_Index': round(np.random.uniform(1, 10), 1),
            'Collaboration_Index': collab_score,
            'Relationship_Issue': rel_reason
        })
    
    df = pd.DataFrame(rows)
    
    # --- ENCODE CATEGORICALS FOR AI ---
    le_country = LabelEncoder()
    df['Country_Code'] = le_country.fit_transform(df['Country'])
    
    le_product = LabelEncoder()
    df['Product_Code'] = le_product.fit_transform(df['Product'])
    
    # Train Model (Now using Product Code!)
    features = ['Country_Risk_Index', 'Order_Value_USD', 'Capacity_Utilization', 'Country_Code', 'Product_Code']
    clf = RandomForestClassifier(n_estimators=50, max_depth=7).fit(df[features], df['Is_Late'])
    
    return df, clf, le_country, le_product

df, model, le_country, le_product = get_data()

# --- 3. TOP NAVIGATION ---
st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
c_title, c_f1, c_f2, c_f3, c_f4,c_f5, c_stat = st.columns([3, 1, 1, 1, 1, 0.8, 0.8])

with c_title:
    st.title("Supplier Risk Analysis")
    # st.caption("Enterprise Risk Dashboard")

with c_f1:
    all_regions = sorted(df['Region'].unique())
    sel_region = st.multiselect("Region", all_regions, default=[])

with c_f2:
    if sel_region:
        filter_1 = df[df['Region'].isin(sel_region)]
        available_countries = sorted(filter_1['Country'].unique())
    else:
        available_countries = sorted(df['Country'].unique())
        
    sel_country = st.multiselect("Country", available_countries, default=[])

with c_f3:
    # New Supplier Filter
    temp_df = df.copy()
    # Apply Region filter if selected
    if sel_region: temp_df = temp_df[temp_df['Region'].isin(sel_region)]
    # Apply Country filter if selected
    if sel_country: temp_df = temp_df[temp_df['Country'].isin(sel_country)]

    available_vendors = sorted(temp_df['Vendor'].unique())
    sel_vendor = st.multiselect("Supplier", available_vendors, default=[])

with c_f4:
    if sel_vendor: temp_df = temp_df[temp_df['Vendor'].isin(sel_vendor)]
    
    available_modes = sorted(temp_df['Shipment_Mode'].unique())
    sel_mode = st.multiselect("Mode", available_modes, default=[])


# Apply Filters
filtered_df = df.copy()
if sel_region: filtered_df = filtered_df[filtered_df['Region'].isin(sel_region)]
if sel_country: filtered_df = filtered_df[filtered_df['Country'].isin(sel_country)]
if sel_mode: filtered_df = filtered_df[filtered_df['Shipment_Mode'].isin(sel_mode)]

with c_f5:
    if sel_mode: temp_df = temp_df[temp_df['Shipment_Mode'].isin(sel_mode)]
    # Custom HTML to match the style of input boxes (Label on top, Box below)
    st.markdown(f"""
        <div style="font-size: 0.85rem; color: #31333F; margin-bottom: 5px;">Total Orders</div>
        <div style="
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 8px 10px;
            color: #31333F;
            font-size: 1rem;
            font-weight: 600;
            text-align: center;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        ">
            {len(filtered_df):,}
        </div>
    """, unsafe_allow_html=True)
base_df = df.copy()
if sel_region: base_df = base_df[base_df['Region'].isin(sel_region)]
if sel_country: base_df = base_df[base_df['Country'].isin(sel_country)]
if sel_mode: base_df = base_df[base_df['Shipment_Mode'].isin(sel_mode)]

# 2. Dependency Calculation (Must happen BEFORE filtering by supplier)
spend_base = base_df.groupby('Vendor')['Order_Value_USD'].sum()
concentration = (spend_base.max() / spend_base.sum()) * 100 if spend_base.sum() > 0 else 0

top_supplier_name = "N/A"
if not spend_base.empty and spend_base.sum() > 0:
    top_supplier_name = spend_base.idxmax()

# 3. Final Filter (Apply Supplier Selection HERE)
# This 'filtered_df' is what ALL the charts below must use
filtered_df = base_df.copy()

if sel_vendor: 
    filtered_df = filtered_df[filtered_df['Vendor'].isin(sel_vendor)]

# Debug Check (Optional but helpful)
if filtered_df.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
    st.stop()

# with c_stat:
#     st.markdown(f"""
#         <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
#             <div class="status-badge">
#                 <span style="color: #0066cc;">System Live</span><br>
#                 <span style="font-size: 0.8rem; color: #555;">{len(filtered_df):,} Orders</span>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)

# st.divider()

# --- 4. MAIN TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌍 External Risk", "💰 Financial", "⚙️ Ops Performance", "✅ Quality & ESG", "🤝 Relationship", "🤖 Risk Simulator"
])

# === TAB 1: EXTERNAL ===
with tab1:
    st.markdown("External & Geopolitical Risk")
    k1, k2, k3 = st.columns(3)
    
    avg_risk = filtered_df['Country_Risk_Index'].mean()
    if avg_risk > 50:
        geo_delta_color = "inverse"  # Red for bad (above target)
    else:
        geo_delta_color = "normal"

    climate_danger_orders = filtered_df[filtered_df['Climate_Zone_Detail'] != 'Safe Zone']
    danger_count = len(climate_danger_orders)
    # climate_danger_orders = filtered_df[filtered_df['Climate_Zone_Detail'] != 'Safe Zone']
    high_risk_orders = filtered_df[filtered_df['Country_Risk_Index'] > 40]
    high_risk_count = len(high_risk_orders)
    # spend = filtered_df.groupby('Vendor')['Order_Value_USD'].sum()
    spend_base = base_df.groupby('Vendor')['Order_Value_USD'].sum()
    total_region_spend = spend_base.sum()
    concentration = (spend_base.max() / spend_base.sum()) * 100 if spend_base.sum() > 0 else 0

    # clim_count = len(climate_danger_orders)
    # Calculate Top Supplier Name from Base Data
    top_supplier_name = "N/A"
    if not spend_base.empty and spend_base.sum() > 0:
        top_supplier_name = spend_base.idxmax()
    if danger_count == 0:
        # If 0 orders are in danger, show "Safe" message in Green/Grey
        clim_msg = "Region Safe"
        clim_color = "off" 
    else:
        # If orders are in danger, show "Risk" message in Red
        clim_msg = "High Risk of Disaster"
        clim_color = "inverse"
    if sel_vendor:
        # CASE A: User selected a supplier -> Show THAT supplier's share
        selected_spend = base_df[base_df['Vendor'].isin(sel_vendor)]['Order_Value_USD'].sum()
        concentration = (selected_spend / total_region_spend) * 100 if total_region_spend > 0 else 0
        dep_label = "Selected Supplier Share"
    else:
        # CASE B: No selection -> Show the biggest supplier in the region
        concentration = (spend_base.max() / total_region_spend) * 100 if total_region_spend > 0 else 0
        dep_label = "Dependency on Top Supplier"

    # Color logic for dependency
    dep_color = "inverse" if concentration > 15 else "normal"
    
    
    k1.metric("Geopolitical Instability", f"{avg_risk:.0f}/100", delta="Target < 50", delta_color=geo_delta_color)
    k2.metric("Orders in Climate Danger Zones", f"{danger_count}", delta=clim_msg, delta_color=clim_color)
    # k3.metric("Dependency on Top Supplier", f"{concentration:.1f}%", delta="Single Source Risk", delta_color="inverse" if concentration > 15 else "normal")
    k3.metric(dep_label, f"{concentration:.1f}%", delta="of Total Region Spend", delta_color=dep_color)
    
    # st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Risk by Country") # UPDATED HEADER
        climate_counts = filtered_df['Climate_Zone_Detail'].value_counts().reset_index()
        climate_counts.columns = ['Zone Type', 'Orders']
        country_risk = filtered_df.groupby('Country')['Country_Risk_Index'].mean().reset_index().sort_values('Country_Risk_Index', ascending=False).head(7)
        # fig = px.bar(country_risk, x='Country_Risk_Index', y='Country', orientation='h', color='Country_Risk_Index', title="Top Risky Countries", color_continuous_scale='Reds', template="plotly_white",text_auto=True)
        fig = px.bar(
            country_risk, 
            x='Country_Risk_Index', 
            y='Country', 
            orientation='h', 
            color='Country_Risk_Index', 
            title="", 
            color_continuous_scale='Oranges', 
            text_auto=True,  # Show numbers on bars automatically
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'),  # <--- THIS IS KEY: Forces global text to black
            xaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            coloraxis_colorbar=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            )
        )
        st.plotly_chart(fig, use_container_width=True) 
        # st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("Climate") # UPDATED HEADER
        climate_data = high_risk_orders['Climate_Zone_Detail'].value_counts().reset_index()
        climate_data.columns = ['Zone Type', 'Orders']
        # fig2 = px.pie(climate_data, values='Orders', names='Zone Type', hole=0.4, template="plotly_white", title="Danger Zones Distribution")
        fig2 = px.pie(
            climate_data, 
            values='Orders', 
            names='Zone Type', 
            hole=0.4, 
            title="",
            color='Zone Type',
            color_discrete_map={'Safe Zone': '#2ecc71', 'Typhoon Belt': '#e74c3c', 'Flood Plain': '#3498db', 'Wildfire Zone': '#e67e22'}
        )
        fig2.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'), # Forces legend and labels to black
            legend=dict(
                font=dict(color='black')
            )
        )
        # Force the labels inside the pie slices to be readable
        fig2.update_traces(textfont_color='white')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("Affected Products & Locations")
    
    # 1. Prepare Data
    # Use filtered_df to respect the supplier selection
    # Sort by Risk so the bad stuff is at the top
    drill_data = filtered_df.sort_values('Country_Risk_Index', ascending=False).head(15)
    
    # 2. Define Columns to Show
    drill_cols = ['Order_ID', 'Product', 'Country', 'Risk_Reason_Geo', 'Shipment_Mode', 'Country_Risk_Index']
    
    # 3. Create Plotly Table (The "Rich" Interactive Table)
    import plotly.graph_objects as go

    fig_table = go.Figure(data=[go.Table(
        columnorder = [0, 1, 2, 3, 4, 5],
        columnwidth = [80, 100, 80, 120, 80, 80], # Adjust width of columns
        
        header=dict(
            values=[f"<b>{c.replace('_', ' ')}</b>" for c in drill_cols], # Bold Headers
            line_color='black',       # Black Border
            fill_color='white',       # White Background
            align='left',
            font=dict(color='black', size=12, family="Arial Black"), # Black Text
            height=30
        ),
        
        cells=dict(
            values=[drill_data[k] for k in drill_cols],
            line_color='black',       # Black Border
            fill_color='white',       # White Background
            align='left',
            font=dict(color='black', size=12), # Black Text
            height=25
        )
    )])

    # 4. Remove margins so it fits tight
    fig_table.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300 # Fixed height so you can scroll inside it
    )

    st.plotly_chart(fig_table, use_container_width=True)

    # st.dataframe(high_risk_orders[['Order_ID', 'Product', 'Country', 'Region', 'Risk_Reason_Geo', 'Climate_Zone_Detail', 'Country_Risk_Index']].head(10), hide_index=True, use_container_width=True)

# === TAB 2: FINANCIAL (Detailed) ===
with tab2:
    st.markdown("Financial Health")
    k1, k2, k3 = st.columns(3)
    
    bad_credit_df = filtered_df[filtered_df['Supplier_Credit_Rating'].isin(['C', 'CC'])]
    bad_credit_count = len(bad_credit_df)
    avg_dpo = filtered_df['DPO_Impact_Days'].mean()
    avg_cost = filtered_df['Cost_Competitiveness'].mean()
    
    k1.metric("Suppliers with Poor Credit", f"{bad_credit_count}", delta="High Financial Risk", delta_color="inverse")
    k2.metric("Avg Days to Pay Suppliers", f"{avg_dpo:.0f} Days", help="How long we take to pay our vendors")
    k3.metric("Price vs. Market Rate", f"{avg_cost:.1f}%", delta="100% = Market Avg", delta_color="normal")
    
    # st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Why are suppliers risky?")
        fin_reasons_df = bad_credit_df['Financial_Risk_Reason'].value_counts().reset_index()
        fin_reasons_df.columns = ['Reason', 'Count']
        fig = px.bar(fin_reasons_df, x='Count', y='Reason', orientation='h', color='Count', color_continuous_scale='Oranges',text_auto=True)
        # fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font_color='black') 
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'), # Global Black Font
            xaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            coloraxis_colorbar=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        # st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        
        st.markdown("Financial Watchlist")
        # st.dataframe(bad_credit_df[['Vendor', 'Country', 'Product', 'Financial_Risk_Reason', 'DPO_Impact_Days']].head(10), hide_index=True, use_container_width=True)
        # st.markdown("Affected Products & Locations") # UPDATED HEADER
        drill_cols = ['Vendor', 'Country', 'Product', 'Financial_Risk_Reason', 'DPO_Impact_Days']
        
        if not bad_credit_df.empty:
            # 2. Prepare Data (Top 15 for scrolling)
            table_data = bad_credit_df[drill_cols].head(15)
            
            # 3. Create Plotly Table
            import plotly.graph_objects as go

            fig_table = go.Figure(data=[go.Table(
                columnorder = [0, 1, 2, 3, 4],
                columnwidth = [120, 80, 80, 120, 60], # Adjust widths so "Reason" and "Vendor" fit
                
                header=dict(
                    values=[f"<b>{c.replace('_', ' ')}</b>" for c in drill_cols], # Bold Headers
                    line_color='black',       # Black Border
                    fill_color='white',       # White Background
                    align='left',
                    font=dict(color='black', size=12, family="Arial Black"), # Black Text
                    height=30
                ),
                
                cells=dict(
                    values=[table_data[k] for k in drill_cols],
                    line_color='black',       # Black Border
                    fill_color='white',       # White Background
                    align='left',
                    font=dict(color='black', size=12), # Black Text
                    height=25
                )
            )])

            # 4. Remove margins & set height
            fig_table.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=300 # Fixed height for scrolling
            )

            st.plotly_chart(fig_table, use_container_width=True)
            
        else:
            st.info("✅ No high-risk suppliers found in current filter.")


# === TAB 3: OPS (Detailed) ===
with tab3:
    st.markdown("Operational Performance")
    k1, k2, k3, k4 = st.columns(4)
    
    late_df = filtered_df[filtered_df['Is_Late'] == 1]
    otd = (1 - filtered_df['Is_Late'].mean()) * 100
    lt_var = filtered_df['Delay_Days'].std()
    acc = filtered_df['Order_Accuracy_Rate'].mean()
    cap = filtered_df['Capacity_Utilization'].mean()
    
    k1.metric("Shipments On Time", f"{otd:.1f}%", delta="Target 95%", delta_color="normal")
    k2.metric("Delivery Consistency", f"±{lt_var:.1f} Days")
    k3.metric("Error-Free Orders", f"{acc:.1f}%")
    k4.metric("Factory Strain", f"{cap:.0f}%", delta="Strain > 90%", delta_color="inverse" if cap > 90 else "off")
    
    # st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Why are shipments late?")
        ops_reasons_df = late_df['Delay_Root_Cause'].value_counts().reset_index()
        ops_reasons_df.columns = ['Reason', 'Count']
        # fig = px.bar(ops_reasons_df, x='Count', y='Reason', orientation='h', color='Count', color_continuous_scale='Blues')
        fig = px.bar(
            ops_reasons_df, 
            x='Count', 
            y='Reason', 
            orientation='h',          # Horizontal bars are better for long labels
            color='Count',            # Color code by frequency
            color_continuous_scale='Viridis', # Professional color scale
            text_auto=True            # Show numbers on bars
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'), # Global Black Font
            xaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            coloraxis_colorbar=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("Late Orders")
        drill_cols = ['Order_ID', 'Product', 'Country', 'Vendor', 'Delay_Root_Cause', 'Delay_Days']
        
        if not late_df.empty:
            # 2. Prepare Data (Top 15 for scrolling)
            table_data = late_df[drill_cols].head(15)
            
            # 3. Create Plotly Table
            import plotly.graph_objects as go

            fig_table = go.Figure(data=[go.Table(
                columnorder = [0, 1, 2, 3, 4, 5],
                # Adjust column widths (give Vendor and Reason more space)
                columnwidth = [80, 80, 70, 110, 110, 60], 
                
                header=dict(
                    values=[f"<b>{c.replace('_', ' ')}</b>" for c in drill_cols], # Bold Headers
                    line_color='black',       # Black Border
                    fill_color='white',       # White Background
                    align='left',
                    font=dict(color='black', size=12, family="Arial Black"), # Black Text
                    height=30
                ),
                
                cells=dict(
                    values=[table_data[k] for k in drill_cols],
                    line_color='black',       # Black Border
                    fill_color='white',       # White Background
                    align='left',
                    font=dict(color='black', size=12), # Black Text
                    height=25
                )
            )])

            # 4. Remove margins & set height
            fig_table.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=300 # Fixed height for scrolling
            )

            st.plotly_chart(fig_table, use_container_width=True)
            
        else:
            st.info("✅ All orders are On Time in this view.")
# === TAB 4: QUALITY (Detailed) ===
with tab4:
    st.markdown("Quality & Compliance")
    k1, k2, k3, k4 = st.columns(4)
    
    defect_df = filtered_df[filtered_df['Defect_Flag'] == 1]
    defect = filtered_df['Defect_Flag'].mean() * 100
    certified = len(filtered_df[filtered_df['Audit_Status'] == 'Certified'])
    cert_rate = (certified / len(filtered_df)) * 100
    esg = filtered_df['ESG_Score'].mean()
    violations = len(filtered_df[filtered_df['Regulatory_Compliance'] == 'Fail'])
    
    k1.metric("Item Defect Rate", f"{defect:.2f}%", delta="Target < 2%", delta_color="inverse")
    k2.metric("ISO Certified Suppliers", f"{cert_rate:.0f}%")
    k3.metric("Compliance Violations", f"{violations}", delta="Legal Risk" if violations > 0 else "Clean", delta_color="inverse" if violations > 0 else "normal")
    k4.metric("Sustainability Score", f"{esg:.0f}/100")
    
    # st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Why did products fail?")
        qual_reasons_df = defect_df['Defect_Root_Cause'].value_counts().reset_index().head(5)
        qual_reasons_df.columns = ['Reason', 'Count']
        fig = px.bar(
            qual_reasons_df, 
            x='Count', 
            y='Reason', 
            orientation='h',          # Horizontal bars are better for long labels
            color='Count',            # Color code by frequency
            color_continuous_scale='Viridis', # Professional color scale
            text_auto=True            # Show numbers on bars
        )
        # fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font_color='black')
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'), # Global Black Font
            xaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            coloraxis_colorbar=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("Defect & Compliance Log")
        drill_cols = ['Order_ID', 'Product', 'Country', 'Vendor', 'Defect_Root_Cause', 'Regulatory_Compliance']
        
        if not defect_df.empty:
            # 2. Prepare Data (Top 15)
            table_data = defect_df[drill_cols].head(15)
            
            # 3. Create Plotly Table
            import plotly.graph_objects as go

            fig_table = go.Figure(data=[go.Table(
                columnorder = [0, 1, 2, 3, 4, 5],
                # Adjust widths: give more space to Vendor and Root Cause
                columnwidth = [80, 80, 70, 110, 110, 90], 
                
                header=dict(
                    values=[f"<b>{c.replace('_', ' ')}</b>" for c in drill_cols], # Bold Headers
                    line_color='black',       # Black Border
                    fill_color='white',       # White Background
                    align='left',
                    font=dict(color='black', size=12, family="Arial Black"), # Black Text
                    height=30
                ),
                
                cells=dict(
                    values=[table_data[k] for k in drill_cols],
                    line_color='black',       # Black Border
                    fill_color='white',       # White Background
                    align='left',
                    font=dict(color='black', size=12), # Black Text
                    height=25
                )
            )])

            # 4. Remove margins & set height
            fig_table.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=300 
            )

            st.plotly_chart(fig_table, use_container_width=True)
            
        else:
            st.info("✅ No Quality or Compliance issues found.")
# === TAB 5: RELATIONSHIP (Detailed) ===
with tab5:
    st.markdown("Relationship & Responsiveness")
    k1, k2, k3 = st.columns(3)
    
    friction_df = filtered_df[filtered_df['Collaboration_Index'] < 4.0]
    
    resp = filtered_df['Responsiveness_Score'].mean()
    innov = filtered_df['Innovation_Index'].mean()
    collab = filtered_df['Collaboration_Index'].mean()
    
    k1.metric("Communication Speed", f"{resp:.1f}/100", help="Supplier Responsiveness Score")
    k2.metric("Innovation Score", f"{innov:.1f}/10", help="Contribution to new product dev")
    k3.metric("Partnership Strength", f"{collab:.1f}/10", help="Collaboration Index")
    
    # st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Top Friction Points")
        rel_reasons_df = friction_df['Relationship_Issue'].value_counts().reset_index()
        rel_reasons_df.columns = ['Reason', 'Count']
        # fig = px.bar(rel_reasons_df, x='Count', y='Reason', orientation='h', color='Count', color_continuous_scale='Purples')
        fig = px.bar(
            rel_reasons_df, 
            x='Count', 
            y='Reason', 
            orientation='h',          # Horizontal bars are better for long labels
            color='Count',            # Color code by frequency
            color_continuous_scale='Viridis', # Professional color scale
            text_auto=True            # Show numbers on bars
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black'), # Global Black Font
            xaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            ),
            coloraxis_colorbar=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black')
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("Partner Issue Log")
        drill_cols = ['Vendor', 'Country', 'Relationship_Issue', 'Collaboration_Index']
        
        if not friction_df.empty:
            # 2. Prepare Data (Remove duplicates & Take Top 15)
            # We use drop_duplicates() here because relationship issues are usually vendor-level, not order-level
            table_data = friction_df[drill_cols].drop_duplicates().head(15)
            
            # 3. Create Plotly Table
            import plotly.graph_objects as go

            fig_table = go.Figure(data=[go.Table(
                columnorder = [0, 1, 2, 3],
                # Adjust widths: Give 'Relationship Issue' the most space
                columnwidth = [100, 70, 130, 80], 
                
                header=dict(
                    values=[f"<b>{c.replace('_', ' ')}</b>" for c in drill_cols], # Bold Headers
                    line_color='black',       # Black Border
                    fill_color='white',       # White Background
                    align='left',
                    font=dict(color='black', size=12, family="Arial Black"), # Black Text
                    height=30
                ),
                
                cells=dict(
                    values=[table_data[k] for k in drill_cols],
                    line_color='black',       # Black Border
                    fill_color='white',       # White Background
                    align='left',
                    font=dict(color='black', size=12), # Black Text
                    height=25
                )
            )])

            # 4. Remove margins & set height
            fig_table.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=300 
            )

            st.plotly_chart(fig_table, use_container_width=True)
            
        else:
            st.info("✅ All supplier relationships are in good standing.")
# === TAB 6: SIMULATOR (Includes Product Selection) ===
with tab6:
    st.markdown("Risk Simulator")
    st.caption("Predict supply chain disruption probability for new orders.")
    
    c_sim1, c_sim2 = st.columns([1, 2])
    with c_sim1:
        with st.form("sim_form"):
            st.markdown("#### **Order Parameters**") # Bold Header
            
            # The CSS above will force these to be White/Black
            country_in = st.selectbox("Origin Country", sorted(df['Country'].unique()))
            prod_in = st.selectbox("Product Type", sorted(df['Product'].unique()))
            
            r_in = st.slider("Region Risk Index (0-100)", 0, 100, 60)
            v_in = st.number_input("Order Value ($)", min_value=10000, max_value=1000000, value=50000)
            c_in = st.slider("Factory Capacity Utilization (%)", 0, 100, 85)
            
            # The CSS targets 'stFormSubmitButton' to make this Blue
            submitted = st.form_submit_button("Run Prediction", use_container_width=True)
    
    with c_sim2:
        if submitted:
            # Encode Inputs
            ctry_code = le_country.transform([country_in])[0]
            prod_code = le_product.transform([prod_in])[0]
            
            # Predict
            prob = model.predict_proba([[r_in, v_in, c_in, ctry_code, prod_code]])[0][1]
            st.markdown("#### Prediction Result")
            if prob > 0.5:
                st.error(f"🛑 **HIGH RISK: {prob:.1%} Probability of Delay**")
                st.info("Recommendation: Engage backup supplier or increase lead time buffer.")
            else:
                st.success(f"✅ **LOW RISK: {prob:.1%} Probability of Delay**")
                st.info("Recommendation: Proceed with standard procurement process.")
        else:
            st.info("👈 Enter parameters and click 'Run Prediction' to see AI analysis.")
