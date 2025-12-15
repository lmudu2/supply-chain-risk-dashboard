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
    region_map = {
        'Region_US_East': ['USA', 'Canada', 'Mexico', 'Panama', 'Costa Rica'],
        'Region_EU_Central': ['Germany', 'France', 'Poland', 'Italy', 'Netherlands', 'Spain', 'Sweden'],
        'Region_Asia_Pacific': ['China', 'Vietnam', 'Thailand', 'India', 'Japan', 'South Korea', 'Malaysia'],
        'Region_LatAm_South': ['Brazil', 'Argentina', 'Chile', 'Peru', 'Colombia'],
        'Region_Africa_West': ['Nigeria', 'Ghana', 'Ivory Coast', 'Senegal', 'Cameroon']
    }
    regions = list(region_map.keys())

    vendors_by_region = {
        'Region_US_East': [f'US_Supply_Co_{i}' for i in range(1, 6)],
        'Region_EU_Central': [f'Euro_Ind_{i}' for i in range(1, 6)],
        'Region_Asia_Pacific': [f'Asia_Tech_{i}' for i in range(1, 8)],
        'Region_LatAm_South': [f'LatAm_Logistics_{i}' for i in range(1, 5)],
        'Region_Africa_West': [f'Afro_Trade_{i}' for i in range(1, 4)]
    }

    country_product_map = {
        # --- ASIA ---
        'China': ['Consumer Electronics', 'Plastic Resins', 'Steel Rolls', 'Lithium Batteries'],
        'Vietnam': ['Textiles', 'Rubber', 'Coffee Beans', 'Electronics'],
        'Thailand': ['Auto Parts', 'Electronics', 'Rubber'],
        'India': ['Textiles', 'Raw Cotton', 'Spices', 'Steel Rolls'],
        'Japan': ['Auto Parts', 'Microchips', 'Robotics'],
        'South Korea': ['Microchips', 'Consumer Electronics', 'Steel Rolls'],
        'Malaysia': ['Microchips', 'Rubber', 'Palm Oil'],

        # --- EUROPE ---
        'Germany': ['Auto Parts', 'Industrial Machinery', 'Chemicals', 'Steel Rolls'],
        'France': ['Luxury Goods', 'Aerospace Parts', 'Chemicals', 'Wine/Beverage'],
        'Italy': ['Textiles', 'Furniture', 'Leather', 'Industrial Machinery'],
        'Netherlands': ['Chemicals', 'Flowers/Agri', 'Electronics'],
        'Spain': ['Agri Products', 'Auto Parts', 'Textiles'],
        'Sweden': ['Industrial Machinery', 'Auto Parts', 'Steel Rolls'],
        'Poland': ['Furniture', 'Auto Parts', 'Copper'],

        # --- AMERICAS ---
        'USA': ['Medical Devices', 'Chemicals', 'Precision Parts', 'Corn/Wheat'],
        'Canada': ['Wood/Lumber', 'Raw Materials', 'Auto Parts'],
        'Mexico': ['Auto Parts', 'Consumer Electronics', 'Silver', 'Plastic Resins'],
        'Brazil': ['Coffee Beans', 'Iron Ore', 'Raw Cotton', 'Soybeans'],
        'Argentina': ['Soybeans', 'Corn/Wheat', 'Lithium Batteries'],
        'Chile': ['Copper', 'Lithium Batteries', 'Fruit'],
        'Peru': ['Copper', 'Gold/Silver', 'Textiles'],
        'Colombia': ['Coffee Beans', 'Flowers/Agri', 'Oil/Petroleum'],
        'Panama': ['Bananas/Fruit', 'Seafood'],
        'Costa Rica': ['Medical Devices', 'Bananas/Fruit'],

        # --- AFRICA ---
        'Nigeria': ['Crude Oil', 'Cocoa Beans'],
        'Ghana': ['Cocoa Beans', 'Gold/Diamonds'],
        'Ivory Coast': ['Cocoa Beans', 'Coffee Beans', 'Rubber'],
        'Senegal': ['Fish/Seafood', 'Chemicals'],
        'Cameroon': ['Cocoa Beans', 'Wood/Lumber', 'Oil/Petroleum']
    }

    # PRODUCT COMPLEXITY MAP (Risk Adder)
    # Higher number = Higher inherent risk of delay/damage
    product_risk_map = {
        'Microchips': 0.25, 'Medical Devices': 0.20, 'Lithium Batteries': 0.30,
        'Consumer Electronics': 0.15, 'Auto Parts': 0.10, 'Chemicals': 0.20,
        'Textiles': 0.05, 'Coffee Beans': 0.10, 'Cocoa Beans': 0.12,
        'Industrial Machinery': 0.15, 'Luxury Goods': 0.10, 'Aerospace Parts': 0.25,
        'Robotics': 0.20, 'Toys': 0.05, 'Footwear': 0.05, 'Furniture': 0.05,
        'Pharmaceuticals': 0.25, 'Agricultural Products': 0.15,
        'Raw Cotton': 0.08, 'Steel Rolls': 0.05, 'Iron Ore': 0.05, 
        'Plastic Resins': 0.05, 'Rubber': 0.05, 'Gold/Diamonds': 0.30,
        'Crude Oil': 0.10, 'Copper': 0.05
    }

    defect_map = {
        # --- FINISHED GOODS ---
        'Microchips': ['Calibration Error', 'Temperature Excursion', 'Software Bug', 'ESD Damage'],
        'Medical Devices': ['Sterilization Failure', 'Software Bug', 'Calibration Error', 'Packaging Breach'],
        'Lithium Batteries': ['Leakage', 'Overheating Risk', 'Voltage Instability', 'Labeling Error'],
        'Consumer Electronics': ['Software Bug', 'Battery Drain', 'Cosmetic Scratch', 'Dead Pixel'],
        'Auto Parts': ['Dimension Mismatch', 'Rust/Corrosion', 'Material Fatigue', 'Welding Defect'],
        'Textiles': ['Color Mismatch', 'Stitching Defect', 'Fabric Tear', 'Pattern Error'],
        'Furniture': ['Scratch/Dent', 'Wood Warp', 'Missing Part', 'Upholstery Tear'],
        'Luxury Goods': ['Cosmetic Scratch', 'Packaging Breach'],
        'Aerospace Parts': ['Dimension Mismatch', 'Material Fatigue'],
        'Robotics': ['Software Bug', 'Calibration Error'],
        'Toys': ['Broken Component', 'Paint Defect'],
        'Footwear': ['Stitching Defect', 'Material Tear'],
        'Pharmaceuticals': ['Contamination', 'Temperature Excursion'],
        
        # --- RAW MATERIALS (METALS) ---
        'Steel Rolls': ['Rust/Corrosion', 'Thickness Variance', 'Surface Scratch', 'Edge Crack'],
        'Iron Ore': ['Moisture Content High', 'Impurity/Silica', 'Granule Size Fail'],
        'Copper': ['Oxidation', 'Purity Variance'],
        'Gold/Diamonds': ['Fake/Synthetic', 'Weight Discrepancy', 'Certification Missing'],
        'Silver': ['Tarnish/Oxidation', 'Weight Discrepancy'],

        # --- RAW MATERIALS (AGRICULTURE) ---
        'Raw Cotton': ['Moisture Damage', 'Mold/Fungus', 'Fiber Length Fail', 'Pest Infestation'],
        'Coffee Beans': ['Moisture Damage', 'Mold/Fermentation', 'Insect Damage', 'Broken Beans'],
        'Cocoa Beans': ['Mold/Fungus', 'Fermentation Issue', 'Pest Infestation'],
        'Soybeans': ['Moisture Damage', 'Heat Damage', 'Mold'],
        'Corn/Wheat': ['Moisture Damage', 'Pest Infestation', 'Mold'],
        'Flowers/Agri': ['Wilting', 'Temperature Damage', 'Pest Infestation'],

        # --- RAW MATERIALS (CHEMICALS/PLASTICS) ---
        'Plastic Resins': ['Color Mismatch', 'Melting Point Fail', 'Contamination', 'Moisture Damage'],
        'Chemicals': ['Impurity Found', 'Container Leak', 'Concentration Error', 'Labeling Error'],
        'Rubber': ['Hardness Variance', 'Surface Cracking', 'Moisture Damage'],
        'Crude Oil': ['Water Content High', 'Sulfur Content High', 'Density Fail']
    }

    # # Mappings
    # region_map = {
    #     'Region_US_East': ['USA', 'Canada', 'Mexico', 'Panama', 'Costa Rica'],
    #     'Region_EU_Central': ['Germany', 'France', 'Poland', 'Italy', 'Netherlands', 'Spain', 'Sweden'],
    #     'Region_Asia_Pacific': ['China', 'Vietnam', 'Thailand', 'India', 'Japan', 'South Korea', 'Malaysia'],
    #     'Region_LatAm_South': ['Brazil', 'Argentina', 'Chile', 'Peru', 'Colombia'],
    #     'Region_Africa_West': ['Nigeria', 'Ghana', 'Ivory Coast', 'Senegal', 'Cameroon']
    # }
    # regions = list(region_map.keys())

    # Root Causes
    geo_reasons = ['Trade Tariffs', 'Port Strike', 'Civil Unrest', 'Border Closure', 'Sanctions']
    climate_zones = ['Typhoon Belt', 'Flood Plain', 'Wildfire Zone', 'Earthquake Fault', 'Monsoon Region']
    ops_reasons = ['Raw Material Shortage', 'Port Congestion', 'Machinery Breakdown', 'Customs Hold', 'Labor Strike']
    qual_reasons = ['Calibration Error', 'Packaging Damage', 'Impure Raw Material', 'Software Bug', 'Temperature Excursion']
    rel_reasons = ['Slow Response', 'Contract Dispute', 'Management Change', 'Pricing Conflict', 'Lack of Transparency']
    fin_reasons = ['High Debt Ratio', 'Liquidity Crisis', 'Pending Lawsuit', 'Merger Uncertainty', 'Credit Downgrade']


    for i in range(10000):
        # vendor = random.choice(vendors)
        # region = random.choice(regions)
        # country = random.choice(region_map[region])
        # mode = random.choice(modes)
        # product = random.choice(products)

        region = random.choice(regions)
        country = random.choice(region_map[region])
        vendor = random.choice(vendors_by_region[region])
        possible_products = country_product_map.get(country, ['Raw Materials'])
        product = random.choice(possible_products)

        # if region in ['Region_US_East', 'Region_EU_Central']:
        #     mode = random.choice(['Land', 'Air'])
        # else:
        #     mode = random.choice(['Sea', 'Air'])

        if country in ['USA', 'Canada', 'Mexico', 'Panama', 'Costa Rica']:
            mode = random.choice(['Land', 'Air'])
        else:
            # Europe, Asia, Africa, South America -> Sea or Air ONLY
            mode = random.choice(['Sea', 'Air'])

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
        # prob_late += product_risk_map[product]
        prob_late += product_risk_map.get(product, 0.10)

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
        # if defect_flag:
        #     # THIS ENSURES TEXTILES GET TEXTILE DEFECTS
        #     qual_reason = random.choice(defect_map[product])
        # else:
        #     qual_reason = "Perfect"

        audit_status = np.random.choice(['Certified', 'Pending', 'Failed'], p=[0.90, 0.08, 0.02])

        if audit_status == 'Certified':
            # Certified vendors are reliable: Defect rate usually 0-2.5%
            defect_rate = np.random.uniform(0, 0.025)
        elif audit_status == 'Pending':
             # Unverified vendors are riskier: Defect rate 1-4%
            defect_rate = np.random.uniform(0.01, 0.04)
        else: # Failed
            # Failed vendors are bad: Defect rate 3-8%
            defect_rate = np.random.uniform(0.03, 0.08)

        # --- 3. DETERMINE DEFECT FLAG & ROOT CAUSE ---
        # Threshold > 3% is a "Problem Shipment"
        defect_flag = 1 if defect_rate > 0.03 else 0
        
        if defect_flag:
            possible_defects = defect_map.get(product, ['General Defect'])
            qual_reason = random.choice(possible_defects)
        else:
            qual_reason = "Perfect"
            
        # --- 4. COMPLIANCE STATUS ---
        compliance_status = 'Fail' if random.random() < 0.02 else 'Pass'

        # --- 5. RELATIONSHIP METRICS ---
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
            'Audit_Status': audit_status,
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

if 'sel_region_key' not in st.session_state:
    st.session_state.sel_region_key = []
if 'sel_country_key' not in st.session_state:
    st.session_state.sel_country_key = []

country_to_region_map = df[['Country', 'Region']].drop_duplicates().set_index('Country')['Region'].to_dict()

current_countries = st.session_state.sel_country_key
current_regions = st.session_state.sel_region_key

if current_countries:
    # Find the regions for the selected countries
    needed_regions = list(set(country_to_region_map[c] for c in current_countries))

    # If the required Region isn't selected yet, select it!
    # We use a set check to avoid infinite loops
    if not set(needed_regions).issubset(set(current_regions)):
        # Combine existing regions with new ones
        new_regions = list(set(current_regions) | set(needed_regions))
        st.session_state.sel_region_key = new_regions
        # (Optional) If you want the Region update to happen instantly without a second click:
        # st.rerun()


c_title, c_f1, c_f2, c_f3,c_prod, c_f4,c_f5, c_stat = st.columns([2.5, 1, 1, 1, 1, 1, 0.8, 0.8])

with c_title:
    st.title("Supplier Risk Analysis")
    # st.caption("Enterprise Risk Dashboard")

with c_f1:
    all_regions = sorted(df['Region'].unique())
    sel_region = st.multiselect(
        "Region",
        all_regions,
        key='sel_region_key'
    )
with c_f2:
    if sel_region:
        subset = df[df['Region'].isin(sel_region)]
        logical_options = sorted(subset['Country'].unique())
    else:
        logical_options = sorted(df['Country'].unique())

    # Safety Net: Keep current selection visible
    current_selection = st.session_state.sel_country_key
    final_options = sorted(list(set(logical_options) | set(current_selection)))

    sel_country = st.multiselect("Country", final_options, key='sel_country_key')

with c_f3:
    # # New Supplier Filter
    # temp_df = df.copy()
    # # Apply Region filter if selected
    # if sel_region: temp_df = temp_df[temp_df['Region'].isin(sel_region)]
    # # Apply Country filter if selected
    # if sel_country: temp_df = temp_df[temp_df['Country'].isin(sel_country)]

    # available_vendors = sorted(temp_df['Vendor'].unique())
    # sel_vendor = st.multiselect("Supplier", available_vendors, default=[])
    temp_df = df.copy()
    # Filter Vendors by Region/Country first
    if sel_region:
        temp_df = temp_df[temp_df['Region'].isin(sel_region)]
    if sel_country:
        temp_df = temp_df[temp_df['Country'].isin(sel_country)]

    available_vendors = sorted(temp_df['Vendor'].unique())
    sel_vendor = st.multiselect("Supplier", available_vendors, default=[])

with c_prod:
    # Create a temp dataframe just for filtering options
    temp_df = df.copy()
    if sel_vendor:
        temp_df = temp_df[temp_df['Vendor'].isin(sel_vendor)]
    elif sel_country:
        temp_df = temp_df[temp_df['Country'].isin(sel_country)]
    elif sel_region:
        temp_df = temp_df[temp_df['Region'].isin(sel_region)]

    available_products = sorted(temp_df['Product'].unique())
    sel_product = st.multiselect("Product", available_products, default=[])
with c_f4:
    temp_df_mode = df.copy()
    if sel_region: 
        temp_df_mode = temp_df_mode[temp_df_mode['Region'].isin(sel_region)]
    if sel_country: 
        temp_df_mode = temp_df_mode[temp_df_mode['Country'].isin(sel_country)]
    if sel_vendor:
        temp_df_mode = temp_df_mode[temp_df_mode['Vendor'].isin(sel_vendor)]
    if sel_product:
        temp_df_mode = temp_df_mode[temp_df_mode['Product'].isin(sel_product)]

    # 2. Get modes ONLY from this filtered list
    # If "China" is selected, this list will only contain ['Air', 'Sea']
    available_modes = sorted(temp_df_mode['Shipment_Mode'].unique())
    
    sel_mode = st.multiselect("Mode", available_modes, default=[])

# Apply Filters
filtered_df = df.copy()

if sel_region:
    filtered_df = filtered_df[filtered_df['Region'].isin(sel_region)]
if sel_country:
    filtered_df = filtered_df[filtered_df['Country'].isin(sel_country)]
if sel_product:
    filtered_df = filtered_df[filtered_df['Product'].isin(sel_product)]
if sel_vendor:
    filtered_df = filtered_df[filtered_df['Vendor'].isin(sel_vendor)]
if sel_mode:
    filtered_df = filtered_df[filtered_df['Shipment_Mode'].isin(sel_mode)]

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
        geo_delta_color = "inverse"  # Red for bad
        geo_msg = f"{avg_risk - 50:.1f} pts Above Target"
    else:
        geo_delta_color = "normal"   # Green for good
        geo_msg = "Within Safe Limits"

    climate_danger_orders = filtered_df[filtered_df['Climate_Zone_Detail'] != 'Safe Zone']
    danger_count = len(climate_danger_orders)
    total_count = len(filtered_df)

    danger_pct = (danger_count / total_count * 100) if total_count > 0 else 0

    if danger_count == 0:
        clim_msg = "Region Safe"
        clim_color = "off"
    else:
        clim_msg = f"{danger_pct:.1f}% of Vol in Danger" # Better Context
        clim_color = "inverse"

    high_risk_orders = filtered_df[filtered_df['Country_Risk_Index'] > 40]
    high_risk_count = len(high_risk_orders)
    # spend = filtered_df.groupby('Vendor')['Order_Value_USD'].sum()


    spend_base = base_df.groupby('Vendor')['Order_Value_USD'].sum()
    total_region_spend = spend_base.sum()

    top_supplier_name = spend_base.idxmax() if not spend_base.empty else "N/A"

    if sel_vendor:
        # CASE A: User selected a supplier -> Show THAT supplier's share
        selected_spend = base_df[base_df['Vendor'].isin(sel_vendor)]['Order_Value_USD'].sum()
        concentration = (selected_spend / total_region_spend) * 100 if total_region_spend > 0 else 0
        dep_label = "Selected Supplier Share"
        dep_tooltip = "Percentage of total region spend allocated to the currently selected supplier(s)."
    else:
        # CASE B: No selection -> Show the biggest supplier in the region
        concentration = (spend_base.max() / total_region_spend) * 100 if total_region_spend > 0 else 0
        dep_label = "Dependency on Top Supplier"
        dep_tooltip = f"Highest dependency risk. Currently, your top supplier is **{top_supplier_name}**."

    # Color logic for dependency
    dep_color = "inverse" if concentration > 15 else "normal"


    k1.metric(
        label="Geopolitical Risk Score",
        value=f"{avg_risk:.1f} / 100",
        delta=geo_msg,
        delta_color=geo_delta_color,
        help="""
        **SCORE DEFINITION:**
        • 0-40: ✅ Stable (Safe)
        • 41-70: ⚠️ Unstable (Monitor Closely)
        • 71-100: 🛑 Critical (Active Conflict/Tariffs)

        **SOURCE:** Aggregated from Political Stability Index & Tariff Data.
        """
    )

    k2.metric(
        label="Orders in Climate Danger Zones",
        value=f"{danger_count:,}",
        delta=clim_msg,
        delta_color=clim_color,
        help="""
        **RISK DEFINITION:**
        Count of orders passing through high-risk zones (Typhoon Belts, Flood Plains).

        **IMPACT:**
        Orders in these zones have a **30% higher probability** of weather-related delays.
        """
    )
    # k3.metric("Dependency on Top Supplier", f"{concentration:.1f}%", delta="Single Source Risk", delta_color="inverse" if concentration > 15 else "normal")
    k3.metric(
        label=dep_label,
        value=f"{concentration:.1f}%",
        delta="of Total Region Spend",
        delta_color=dep_color,
        help=f"""
        **DEFINITION:**
        {dep_tooltip}

        **BENCHMARK:**
        Ideally, no single supplier should handle more than **20%** of your total volume to prevent "Vendor Lock-in."
        """
    )
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

    st.markdown("Top 15 Affected Products & Locations")

    # 1. Prepare Data
    # Use filtered_df to respect the supplier selection
    # Sort by Risk so the bad stuff is at the top
    drill_data = filtered_df.sort_values('Country_Risk_Index', ascending=False).head(15)

    # 2. Define Columns to Show
    drill_cols = ['Order_ID', 'Product', 'Country', 'Risk_Reason_Geo', 'Vendor', 'Climate_Zone_Detail', 'Shipment_Mode', 'Country_Risk_Index']

    # 3. Create Plotly Table (The "Rich" Interactive Table)
    import plotly.graph_objects as go

    fig_table = go.Figure(data=[go.Table(
        columnorder = [0, 1, 2, 3, 4, 5, 6, 7],
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
        height=300,
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_table, use_container_width=True)

    # st.dataframe(high_risk_orders[['Order_ID', 'Product', 'Country', 'Region', 'Risk_Reason_Geo', 'Climate_Zone_Detail', 'Country_Risk_Index']].head(10), hide_index=True, use_container_width=True)

# === TAB 2: FINANCIAL (Detailed) ===
with tab2:
    st.markdown("Financial Health")
    k1, k2, k3 = st.columns(3)

    bad_credit_df = filtered_df[filtered_df['Supplier_Credit_Rating'].isin(['C', 'CC', 'CCC'])] # Added CCC for completeness
    risky_order_count = len(bad_credit_df)

    if risky_order_count == 0:
        credit_msg = "No Exposure"
        credit_color = "off"
    else:
        credit_msg = "Orders at Risk"
        credit_color = "inverse" # Red
    bad_credit_count = len(bad_credit_df)

    avg_dpo = filtered_df['DPO_Impact_Days'].mean()
    if avg_dpo > 75:
        dpo_msg = "Straining Suppliers (Too Slow)"
        dpo_color = "inverse" # Red warning
    elif avg_dpo < 45:
        dpo_msg = "Hurting Cash Flow (Too Fast)"
        dpo_color = "normal" # Green/Neutral (Depending on preference)
    else:
        dpo_msg = "Healthy Standard"
        dpo_color = "off" # Grey


    avg_cost = filtered_df['Cost_Competitiveness'].mean()
    cost_delta = avg_cost - 100

    if avg_cost > 100:
        cost_msg = f"{cost_delta:.1f}% Overpaying"
        cost_color = "inverse" # Red (Bad)
    else:
        cost_msg = f"{abs(cost_delta):.1f}% Discount"
        cost_color = "normal" # Green (Good)

    k1.metric(
        label="Orders from Risky Suppliers",
        value=f"{risky_order_count:,}",
        delta=credit_msg,
        delta_color=credit_color,
        help="""
        **DEFINITION:**
        Total count of orders assigned to vendors with 'C' or 'CC' credit ratings.

        **RISK:**
        These vendors have a high probability of bankruptcy or operational failure.
        """
    )

    k2.metric(
        label="Avg Invoices",
        value=f"{avg_dpo:.0f} Days",
        delta=dpo_msg,
        delta_color=dpo_color,
        help="""
        **DEFINITION:**
        Average number of days we take to pay our invoices.

        **BENCHMARKS:**
        • < 45 Days: Good for vendor, bad for our cash.
        • 45-75 Days: Healthy industry standard.
        • > 75 Days: Risk of damaging vendor relationships.
        """
    )
    k3.metric(
        label="Price vs. Market Rate",
        value=f"{avg_cost:.1f}%",
        delta=cost_msg,
        delta_color=cost_color,
        help="""
        **DEFINITION:**
        Compares our purchase price to the global market average.

        **SCORING:**
        • 100%: We pay the exact market rate.
        • > 100%: We are **overpaying** (Negative Variance).
        • < 100%: We secured a **discount** (Positive Variance).
        """
    )

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

        st.markdown("Top 10 High-Risk Vendor Watchlist")
        # st.dataframe(bad_credit_df[['Vendor', 'Country', 'Product', 'Financial_Risk_Reason', 'DPO_Impact_Days']].head(10), hide_index=True, use_container_width=True)
        # st.markdown("Affected Products & Locations") # UPDATED HEADER
        watchlist_cols = ['Vendor', 'Product', 'Supplier_Credit_Rating', 'Financial_Risk_Reason', 'DPO_Impact_Days']
        display_cols = ['Vendor', 'Product', 'Rating', 'Main Issue', 'Days to Pay']

        # 2. FILTER DATA (Risky Suppliers Only)
        # We save this into 'financial_risk_df' (NOT 'table_data')
        financial_risk_df = filtered_df[
            filtered_df['Supplier_Credit_Rating'].isin(['C', 'CC', 'CCC', 'BB'])
        ].sort_values(by='Order_Value_USD', ascending=False).head(10)

        if not financial_risk_df.empty:
            # 3. PREPARE TABLE VALUES
            # We must use 'financial_risk_df' here because 'table_data' does not exist in this tab
            table_values = [financial_risk_df[col] for col in watchlist_cols]

            import plotly.graph_objects as go

            fig_fin = go.Figure(data=[go.Table(
                columnorder=[0, 1, 2, 3, 4],
                columnwidth=[100, 100, 60, 120, 80],

                header=dict(
                    values=[f"<b>{c}</b>" for c in display_cols],
                    line_color='black', fill_color='white', align='left',
                    font=dict(color='black', size=12, family="Arial Black"), height=30
                ),

                cells=dict(
                    values=table_values,
                    line_color='black', fill_color='white', align='left',
                    font=dict(color='black', size=12), height=25,
                    
                    # 4. COLOR LOGIC
                    # We reference 'financial_risk_df' here to avoid the NameError
                    font_color=[
                        'black', # Vendor
                        'black', # Product
                        ['red' if 'C' in x else 'black' for x in financial_risk_df['Supplier_Credit_Rating']],
                        'black', # Reason
                        'black'  # DPO
                    ]
                )
            )])

            fig_fin.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300,paper_bgcolor='white')
            st.plotly_chart(fig_fin, use_container_width=True)
        else:
            st.success("✅ Financial Stability: No high-risk vendors detected in this filter.")


# === TAB 3: OPS (Detailed) ===
with tab3:
    st.markdown("Operational Performance")
    k1, k2, k3, k4 = st.columns(4)

    late_df = filtered_df[filtered_df['Is_Late'] == 1]
    # Metric A: On-Time Pct
    on_time_pct = (1 - filtered_df['Is_Late'].mean()) * 100
    if on_time_pct < 90:
        ot_color = "inverse" # Red (Below standard)
        ot_msg = "Needs Improvement"
    else:
        ot_color = "normal" # Green
        ot_msg = "World Class"

    # otd = (1 - filtered_df['Is_Late'].mean()) * 100
    # Metric B: Consistency
    consistency_std = filtered_df['Delay_Days'].std()
    if pd.isna(consistency_std): consistency_std = 0

    # FIX 1: Relaxed the threshold from 5 to 8 so it isn't ALWAYS red
    if consistency_std > 8:
        cons_color = "inverse" # Red (High variance)
        cons_msg = "Unpredictable"
    elif consistency_std > 4:
        cons_color = "off" # Yellow/Grey (Moderate)
        cons_msg = "Variable"
    else:
        cons_color = "normal" # Green (Reliable)
        cons_msg = "Stable Flow"
    # lt_var = filtered_df['Delay_Days'].std()
    # acc = filtered_df['Order_Accuracy_Rate'].mean()

    # Metric C: Accuracy
    avg_accuracy = filtered_df['Order_Accuracy_Rate'].mean()
    if avg_accuracy < 98:
        acc_color = "inverse" # Red
        acc_msg = "Quality Issues"
    else:
        acc_color = "normal"
        acc_msg = "High Accuracy"

    # cap = filtered_df['Capacity_Utilization'].mean()
    # Metric D: Utilization
    avg_util = filtered_df['Capacity_Utilization'].mean()

    if avg_util > 85:
        util_msg = "Overloaded (Risk)"
        util_color = "inverse" # Red
    elif avg_util < 50:
        util_msg = "Underutilized"
        util_color = "off" # Grey
    else:
        util_msg = "Optimal Capacity"
        util_color = "normal" # Green

    k1.metric(
        label="Shipments On Time",
        value=f"{on_time_pct:.1f}%",
        delta=ot_msg,
        delta_color=ot_color,
        help="""
        **DEFINITION:**
        Percentage of orders arriving on or before the committed date.

        **TARGET:**
        > 95% is considered standard for reliable supply chains.
        """
    )

    k2.metric(
        label="Delivery Consistency",
        value=f"±{consistency_std:.1f} Days",
        delta=cons_msg,
        delta_color=cons_color,
        help="""
        **DEFINITION:**
        This is the **Standard Deviation** (the "Spread") of your delivery times.
        
        **WHAT THIS TELLS YOU:**
        It means **68% of your orders** arrive within this ±7 day window.
        
        **NOTE:**
        This is NOT the maximum delay. Extreme outliers (e.g., 20+ days late) are shown in the **"Top 10 Critical Delays"** table below.
        """
    )
    k3.metric(
        label="Error-Free Orders",
        value=f"{avg_accuracy:.1f}%",
        delta=acc_msg,
        delta_color=acc_color,
        help="""
        **DEFINITION:**
        The "Perfect Order Rate" - Orders with correct documentation, correct quantity, and no damage.
        """
    )
    k4.metric(
        label="Factory Strain",
        value=f"{avg_util:.0f}%",
        delta=util_msg,
        delta_color=util_color,
        help="""
        **DEFINITION:**
        Capacity Utilization of the supplier's manufacturing lines.

        **ZONES:**
        • < 60%: Underutilized.
        • 70-85%: ✅ Optimal Health.
        • > 85%: ⚠️ Overloaded (High risk of delay or defects).
        """
    )
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
        st.markdown("Top 10 Late Orders")
        drill_cols = ['Order_ID', 'Product', 'Country', 'Vendor', 'Delay_Root_Cause', 'Delay_Days']

        if not late_df.empty:
            # 2. Prepare Data (Top 15 for scrolling)
            table_data = late_df[drill_cols]\
                .sort_values('Delay_Days', ascending=False)\
                .head(10)

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
                height=300,
                paper_bgcolor='white'
            )

            st.plotly_chart(fig_table, use_container_width=True)

        else:
            st.info("✅ All orders are On Time in this view.")
# === TAB 4: QUALITY (Detailed) ===
with tab4:
    st.markdown("Quality & Compliance")
    k1, k2, k3, k4 = st.columns(4)

    # defect_df = filtered_df[filtered_df['Defect_Flag'] == 1]
    defect = filtered_df['Defect_Flag'].mean() * 100
    
    # Metric A: Defect Rate (Percentage of orders flagged as defective)
    defect_df = filtered_df[filtered_df['Defect_Flag'] == 1]
    defect_pct = filtered_df['Defect_Flag'].mean() * 100
    
    if defect_pct > 10:
        def_msg = "Critical Failure"
        def_color = "inverse"        # Red
    elif defect_pct > 5:
        def_msg = "Warning Area"     # 5-10% is Yellow/Warning
        def_color = "off"            # Grey/Yellow
    else:
        def_msg = "Quality Good"     # < 5% is Green
        def_color = "normal"

    # Metric B: ISO Certification
    certified_count = len(filtered_df[filtered_df['Audit_Status'] == 'Certified'])
    cert_rate = (certified_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    
    if cert_rate < 80:
        cert_msg = "Many Uncertified"
        cert_color = "inverse" # Red
    else:
        cert_msg = "Verified Supply"
        cert_color = "normal"

    # Metric C: Compliance Violations (Count)
    violations = len(filtered_df[filtered_df['Regulatory_Compliance'] == 'Fail'])
    
    if violations > 0:
        viol_msg = "Legal Issues Found"
        viol_color = "inverse" # Red
    else:
        viol_msg = "Clean"
        viol_color = "normal"

    # Metric D: ESG Score
    esg = filtered_df['ESG_Score'].mean()
    k1.metric(
        label="Problem Shipment Rate",
        value=f"{defect_pct:.1f}%",
        delta=def_msg,
        delta_color=def_color,
        help="""
        **DEFINITION:**
        Percentage of shipments that triggered a defect claim (Damage > 3%).
        
        **WHY IS THIS HIGH?**
        Our threshold is strict. Any shipment with >3% broken items is flagged as a "Problem Shipment."
        """
    )
    k2.metric(
        label="ISO Certified Volume",
        value=f"{cert_rate:.0f}%",
        delta=cert_msg,
        delta_color=cert_color,
        help="""
        **DEFINITION:**
        Percentage of orders coming from suppliers with active **ISO 9001** certification.
        
        """
    )
    k3.metric(
        label="Compliance Violations",
        value=f"{violations}",
        delta=viol_msg,
        delta_color=viol_color,
        help="""
        **DEFINITION:**
        Total count of orders flagged for Regulatory Failures (e.g., Missing SDS, Labor Violations).
        
        **TARGET:**
        Target is **0**. Even one violation can lead to customs seizures.
        """
    )
    k4.metric(
        label="Sustainability Score",
        value=f"{esg:.0f}/100",
        delta="Avg ESG Rating",
        delta_color="off",
        help="""
        **DEFINITION:**
        Average Environmental, Social, and Governance (ESG) score of selected suppliers.
        """
    )

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
        st.markdown("Top 10 Defect & Compliance Log")
        
        drill_cols = ['Order_ID', 'Product', 'Country', 'Vendor', 'Defect_Root_Cause', 'Regulatory_Compliance']
        display_names = ['Order ID', 'Product', 'Country', 'Vendor', 'Defect', 'Compliance']
        
        quality_issues_df = filtered_df[
            (filtered_df['Defect_Flag'] == 1) | 
            (filtered_df['Regulatory_Compliance'] == 'Fail')
        ].copy()

        if not quality_issues_df.empty:
            # 3. SMART LOGIC FUNCTION (The Fix)
            def determine_issue(row):
                # If the text says "Perfect" BUT Compliance Failed -> It is actually a Regulatory Failure
                if row['Defect_Root_Cause'] == "Perfect" and row['Regulatory_Compliance'] == 'Fail':
                    return "Regulatory Failure"
                
                # Otherwise, show the actual defect (e.g., "Impurity Found")
                return row['Defect_Root_Cause']

            # 4. APPLY LOGIC (Creates the new column)
            quality_issues_df['Primary_Issue'] = quality_issues_df.apply(determine_issue, axis=1)

            # 5. SORT: Fails first
            table_data = quality_issues_df.sort_values(
                by=['Regulatory_Compliance', 'Defect_Flag'], 
                ascending=[True, False]
            ).head(10)

            table_data_display = table_data[drill_cols]

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
                height=300,
                paper_bgcolor='white'
            )

            st.plotly_chart(fig_table, use_container_width=True)

        else:
            st.info("✅ No Quality or Compliance issues found.")
# === TAB 5: RELATIONSHIP (Detailed) ===
with tab5:
    st.markdown("Relationship & Responsiveness")
    k1, k2, k3 = st.columns(3)

    friction_df = filtered_df[filtered_df['Collaboration_Index'] < 4.0]

    # Metric A: Responsiveness (Communication Speed)
    resp = filtered_df['Responsiveness_Score'].mean()

    if resp > 90:
        resp_msg = "Avg < 2 hrs"
        resp_color = "normal" # Green
    elif resp < 70:
        resp_msg = "Slow (Days)"
        resp_color = "inverse" # Red
    else:
        resp_msg = "Standard (24h)"
        resp_color = "off" # Grey

    # Metric B: Innovation Index
    innov = filtered_df['Innovation_Index'].mean()

    if innov > 7:
        inn_msg = "Strategic Partner" # High value add
        inn_color = "normal"
    elif innov < 3:
        inn_msg = "Passive Maker" # Just takes orders
        inn_color = "off"
    else:
        inn_msg = "Transactional"
        inn_color = "off"

    # Metric C: Partnership Strength (Collaboration)
    collab = filtered_df['Collaboration_Index'].mean()

    # Logic: Low collaboration = High Friction
    if collab < 4:
        col_msg = "High Friction"
        col_color = "inverse" # Red
    elif collab > 8:
        col_msg = "Highly Aligned"
        col_color = "normal" # Green
    else:
        col_msg = "Stable"
        col_color = "off"

    # --- 3. DISPLAY METRICS ---
    k1, k2, k3 = st.columns(3)

    k1.metric(
        label="Communication Speed",
        value=f"{resp:.1f} / 100",
        delta=resp_msg,
        delta_color=resp_color,
        help="""
        **DEFINITION:**
        Measures average speed in acknowledging orders and email replies.
        • **90+:** Same-day response.
        • **<70:** Delayed response.
        """
    )

    k2.metric(
        label="Innovation Score",
        value=f"{innov:.1f} / 10",
        delta=inn_msg,
        delta_color=inn_color,
        help="""
        **DEFINITION:**
        Supplier's contribution to improvements.
        • **1-3:** Passive (No feedback).
        • **8-10:** Strategic (Suggests better designs/costs).
        """
    )

    k3.metric(
        label="Partnership Strength",
        value=f"{collab:.1f} / 10",
        delta=col_msg,
        delta_color=col_color,
        help="""
        **DEFINITION:**
        "Ease of Doing Business" score.
        Scores **< 4.0** indicate frequent disputes or lack of transparency.
        """
    )

    # st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Collaboration Problems")
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
        st.markdown("Top 10 Partner Issue Logs")
        drill_cols = ['Vendor', 'Country', 'Relationship_Issue', 'Collaboration_Index']

        if not friction_df.empty:
            # 2. Prepare Data (Remove duplicates & Take Top 15)
            # We use drop_duplicates() here because relationship issues are usually vendor-level, not order-level
            table_data = friction_df[drill_cols]\
                .sort_values('Collaboration_Index', ascending=True)\
                .drop_duplicates()\
                .head(10)

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
                height=300,
                paper_bgcolor='white'
            )

            st.plotly_chart(fig_table, use_container_width=True)

        else:
            st.info("✅ All supplier relationships are in good standing.")
# === TAB 6: SIMULATOR (Includes Product Selection) ===
# === TAB 6: SIMULATOR (Smart Update) ===
with tab6:
    st.markdown("### 🤖 Risk Simulator")
    st.caption("Predict supply chain disruption probability for new orders.")

    c_sim1, c_sim2 = st.columns([1, 2])
    with c_sim1:
        # --- STEP 1: MOVE SELECTORS OUTSIDE THE FORM ---
        # This allows the app to 'refresh' the risk slider immediately when you change country
        st.markdown("Select Context")
        country_in = st.selectbox("Origin Country", sorted(df['Country'].unique()))

        valid_products = sorted(df[df['Country'] == country_in]['Product'].unique())
        prod_in = st.selectbox("Product Type", valid_products)

        # --- STEP 2: CALCULATE DEFAULTS ---
        # Look up the actual average risk for this country in our data
        current_risk_avg = int(df[df['Country'] == country_in]['Country_Risk_Index'].mean())

        # Look up the Product Risk (just for reference/defaults if needed)
        # (Optional: You could also adjust defaults based on product, but Country is key for Risk Index)

        # --- STEP 3: THE FORM (Inputs that use the defaults) ---
        with st.form("sim_form"):
            st.markdown("Adjust Parameters")

            # The Slider now defaults to 'current_risk_avg'
            # We use 'value=current_risk_avg' so it auto-updates!
            r_in = st.slider("Region Risk Index (0-100)", 0, 100, value=current_risk_avg, help=f"Average Risk for {country_in} is {current_risk_avg}")

            v_in = st.number_input("Order Value ($)", min_value=10000, max_value=1000000, value=50000)
            c_in = st.slider("Factory Capacity Utilization (%)", 0, 100, 85)

            submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    with c_sim2:
        if submitted:
            # Encode Inputs
            ctry_code = le_country.transform([country_in])[0]
            prod_code = le_product.transform([prod_in])[0]

            # Predict
            # The model uses the SLIDER value (r_in), which is now accurate to the country
            prob = model.predict_proba([[r_in, v_in, c_in, ctry_code, prod_code]])[0][1]

            st.markdown("#### 📊 Prediction Result")

            # Create a nice result card
            if prob > 0.5:
                st.markdown(f"""
                    <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4444;">
                        <h3 style="color: #cc0000; margin:0;">🛑 HIGH RISK</h3>
                        <p style="font-size: 1.2rem; font-weight: bold; margin: 5px 0;">Probability of Delay: {prob:.1%}</p>
                        <p style="color: #333;">The combination of <b>{country_in}</b> (Risk: {r_in}) and these order parameters suggests a high likelihood of disruption.</p>
                        <hr>
                        <b>Recommendation:</b> Engage backup supplier or increase lead time buffer.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #00c851;">
                        <h3 style="color: #007e33; margin:0;">✅ LOW RISK</h3>
                        <p style="font-size: 1.2rem; font-weight: bold; margin: 5px 0;">Probability of Delay: {prob:.1%}</p>
                        <p style="color: #333;">Conditions in <b>{country_in}</b> appear stable for this order size.</p>
                        <hr>
                        <b>Recommendation:</b> Proceed with standard procurement process.
                    </div>
                """, unsafe_allow_html=True)

            # Feature Explanation (Optional: Shows why)
            st.caption(f"Model Inputs: Risk={r_in}, Value=${v_in:,}, Cap={c_in}%")

        else:
            # Placeholder before they click run
            st.info(f"👈 Current Average Risk for **{country_in}** is **{current_risk_avg}**. Click 'Run Prediction' to analyze.")
with st.expander("How are these metrics calculated?"):
    st.markdown("""
    ### External Risk
    | Metric | Formula | Interpretation |
    | :--- | :--- | :--- |
    | **Geopolitical Risk** | `Average(Country_Risk_Index)` | **0-40**: Safe, **>70**: Critical Instability (War, Tariffs). |
    | **Climate Danger** | `Count(Orders in 'Typhoon Belt' or 'Flood Plain')` | Orders with a 30% higher probability of weather delay. |
    | **Supplier Dependency** | `(Spend with Top Supplier / Total Spend) * 100` | **>20%** indicates "Vendor Lock-in" risk. |

    ### Financial Health
    | Metric | Formula | Interpretation |
    | :--- | :--- | :--- |
    | **Risky Suppliers** | `Count(Suppliers with Credit Rating 'C' or 'CC')` | Number of active orders with suppliers near bankruptcy. |
    | **Avg Invoice** | `Average(Payment Terms in Days)` | **<45**: We pay too fast (Cash flow hurt). **>75**: We pay too slow (Supplier hurt). |
    | **Price vs Market** | `Average(Purchase Price / Market Rate) * 100` | **100%**: Fair Market Value. **>100%**: Overpaying. **<100%**: Discount. |

    ### Operations
    | Metric | Formula | Interpretation |
    | :--- | :--- | :--- |
    | **On-Time Rate** | `(1 - Late_Rate) * 100` | Percentage of orders arriving on or before the promised date. |
    | **Consistency** | `StdDev(Actual_Date - Promised_Date)` | **±1-3 Days**: Reliable. **>±8 Days**: Highly Unpredictable. |
    | **Error-Free Orders** | `Average(Order_Accuracy_Rate)` | **% Perfect Orders**. Checks for correct quantity, documentation, and packaging. |
    | **Factory Strain** | `Average(Capacity_Utilization)` | **>85%**: Factory is overloaded; rush orders will likely fail. |

    ### Quality & ESG
    | Metric | Formula | Interpretation |
    | :--- | :--- | :--- |
    | **Problem Shipment Rate** | `(Count of Orders with Defects / Total Orders) * 100` | Percentage of shipments arriving with damaged or wrong items. |
    | **ISO Certified** | `(Orders from Certified Vendors / Total Orders) * 100` | **<50%** means we rely heavily on unverified/non-standardized suppliers. |
    | **Compliance Violations** | `Count(Regulatory_Compliance == 'Fail')` | Total count of orders flagged for **missing paperwork**, **safety violations**, or **customs failures**. |
    | **Sustainability Score** | `Average(ESG_Score)` | **0-100 Score**. **<50** indicates high environmental or ethical risk. |

    ### Relationship
    | Metric | Formula | Interpretation |
    | :--- | :--- | :--- |
    | **Communication Speed** | `Average(Responsiveness_Score)` | **0-100 Scale**. Measures speed of email replies/order acknowledgments. **<70** is poor. |
    | **Innovation Score** | `Average(Innovation_Index)` | **1-10 Scale**. Higher score means the supplier proactively suggests design/cost improvements. |
    | **Partnership Strength** | `Average(Collaboration_Index)` | **1-10 Scale**. **<4** indicates frequent friction (disputes, lack of transparency). |
    """)
