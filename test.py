import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import legendre, jv, yn
from scipy.optimize import fsolve
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Mathematical Functions Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #F18F01;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🔬 Advanced Mathematical Functions Dashboard</h1>', unsafe_allow_html=True)

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("viridis")

# Sidebar controls
st.sidebar.markdown("## 🎛️ Control Panel")

# Fibonacci controls
st.sidebar.markdown("### Fibonacci Sequence")
fib_n = st.sidebar.slider("Number of Fibonacci terms", 5, 50, 20)
fib_style = st.sidebar.selectbox("Fibonacci Plot Style", ["line", "bar", "scatter"])

# Legendre polynomial controls
st.sidebar.markdown("### Legendre Polynomials")
leg_degree = st.sidebar.slider("Maximum polynomial degree", 1, 10, 5)
leg_x_range = st.sidebar.slider("X-axis range", 0.5, 3.0, 2.0)
leg_resolution = st.sidebar.slider("Resolution", 100, 1000, 500)

# Bessel function controls
st.sidebar.markdown("### Bessel Functions")
bessel_order = st.sidebar.slider("Bessel function order", 0, 5, 2)
bessel_x_max = st.sidebar.slider("Maximum x value", 10, 50, 20)
bessel_type = st.sidebar.selectbox("Bessel function type", ["First kind (J)", "Second kind (Y)", "Both"])

# Helper functions
@st.cache_data
def generate_fibonacci(n):
    """Generate Fibonacci sequence"""
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib[:n]

@st.cache_data
def generate_legendre_data(max_degree, x_range, resolution):
    """Generate Legendre polynomial data"""
    x = np.linspace(-x_range, x_range, resolution)
    data = {}
    for n in range(max_degree + 1):
        P_n = legendre(n)
        data[f'P_{n}(x)'] = P_n(x)
    return x, data

@st.cache_data
def generate_bessel_data(order, x_max):
    """Generate Bessel function data"""
    x = np.linspace(0.1, x_max, 1000)
    j_data = jv(order, x)
    y_data = yn(order, x)
    return x, j_data, y_data

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Fibonacci Section
    st.markdown('<h2 class="section-header">🌀 Fibonacci Sequence</h2>', unsafe_allow_html=True)
    
    fib_sequence = generate_fibonacci(fib_n)
    fib_df = pd.DataFrame({
        'Index': range(len(fib_sequence)),
        'Value': fib_sequence,
        'Golden Ratio Approximation': [fib_sequence[i]/fib_sequence[i-1] if i > 0 and fib_sequence[i-1] != 0 else 0 for i in range(len(fib_sequence))]
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Fibonacci sequence plot
    if fib_style == "line":
        sns.lineplot(data=fib_df, x='Index', y='Value', ax=ax1, marker='o', linewidth=3, markersize=8)
    elif fib_style == "bar":
        sns.barplot(data=fib_df, x='Index', y='Value', ax=ax1, palette='viridis')
    else:
        sns.scatterplot(data=fib_df, x='Index', y='Value', ax=ax1, s=100, alpha=0.7)
    
    ax1.set_title('Fibonacci Sequence', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Term Index', fontsize=12)
    ax1.set_ylabel('Fibonacci Value', fontsize=12)
    
    # Golden ratio convergence
    sns.lineplot(data=fib_df[1:], x='Index', y='Golden Ratio Approximation', ax=ax2, 
                color='gold', linewidth=3, marker='s', markersize=6)
    ax2.axhline(y=1.618033988749, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Golden Ratio (φ)')
    ax2.set_title('Convergence to Golden Ratio', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Term Index', fontsize=12)
    ax2.set_ylabel('Ratio F(n)/F(n-1)', fontsize=12)
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Statistics panel
    st.markdown("### 📊 Fibonacci Statistics")
    
    if len(fib_sequence) > 1:
        golden_ratio_approx = fib_sequence[-1] / fib_sequence[-2] if fib_sequence[-2] != 0 else 0
        
        st.markdown(f'''
        <div class="metric-container">
            <h4>Last Term</h4>
            <h3>{fib_sequence[-1]:,}</h3>
        </div>
        <div class="metric-container">
            <h4>Golden Ratio Approx</h4>
            <h3>{golden_ratio_approx:.6f}</h3>
        </div>
        <div class="metric-container">
            <h4>Sum of All Terms</h4>
            <h3>{sum(fib_sequence):,}</h3>
        </div>
        ''', unsafe_allow_html=True)

# Legendre Polynomials Section
st.markdown('<h2 class="section-header">📐 Legendre Polynomials</h2>', unsafe_allow_html=True)

x_leg, leg_data = generate_legendre_data(leg_degree, leg_x_range, leg_resolution)

fig, ax = plt.subplots(figsize=(16, 8))
colors = sns.color_palette("husl", len(leg_data))

for i, (label, y_data) in enumerate(leg_data.items()):
    ax.plot(x_leg, y_data, label=label, linewidth=3, color=colors[i], alpha=0.8)

ax.set_title('Legendre Polynomials', fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('P_n(x)', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)

plt.tight_layout()
st.pyplot(fig)

# Bessel Functions Section
st.markdown('<h2 class="section-header">🌊 Bessel Functions</h2>', unsafe_allow_html=True)

x_bessel, j_bessel, y_bessel = generate_bessel_data(bessel_order, bessel_x_max)

col1, col2 = st.columns(2)

with col1:
    if bessel_type in ["First kind (J)", "Both"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot multiple orders for comparison
        for order in range(bessel_order + 1):
            x_temp, j_temp, _ = generate_bessel_data(order, bessel_x_max)
            ax.plot(x_temp, j_temp, label=f'J_{order}(x)', linewidth=3, alpha=0.8)
        
        ax.set_title(f'Bessel Functions of the First Kind', fontsize=16, fontweight='bold')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('J_n(x)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)

with col2:
    if bessel_type in ["Second kind (Y)", "Both"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot multiple orders for comparison
        for order in range(bessel_order + 1):
            x_temp, _, y_temp = generate_bessel_data(order, bessel_x_max)
            ax.plot(x_temp, y_temp, label=f'Y_{order}(x)', linewidth=3, alpha=0.8)
        
        ax.set_title(f'Bessel Functions of the Second Kind', fontsize=16, fontweight='bold')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('Y_n(x)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_ylim(-2, 1)  # Limit y-axis for better visualization
        
        plt.tight_layout()
        st.pyplot(fig)

# 3D Surface Plot Section
st.markdown('<h2 class="section-header">🎯 3D Mathematical Surface</h2>', unsafe_allow_html=True)

# Create a 3D surface combining multiple mathematical functions
x_3d = np.linspace(-5, 5, 50)
y_3d = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x_3d, y_3d)

# Combine Bessel and trigonometric functions for interesting surface
Z = jv(0, np.sqrt(X**2 + Y**2)) * np.exp(-0.1 * (X**2 + Y**2)) * np.cos(X) * np.sin(Y)

fig = go.Figure(data=[go.Surface(
    z=Z, x=X, y=Y,
    colorscale='Viridis',
    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
    colorbar=dict(title="Z Value", titleside="right", len=0.7)
)])

fig.update_layout(
    title='3D Mathematical Surface: J₀(√(x²+y²)) × e^(-0.1(x²+y²)) × cos(x) × sin(y)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    height=600,
    font=dict(size=12)
)

st.plotly_chart(fig, use_container_width=True)

# Footer with instructions
st.markdown("---")
st.markdown("""
### 🚀 **Running the Dashboard**

To run this dashboard on your headless server at `172.30.98.56:7777`, save this code as `math_dashboard.py` and run:

```bash
streamlit run math_dashboard.py --server.address 172.30.98.56 --server.port 7777
```

**Required packages:**
```bash
pip install streamlit numpy pandas seaborn matplotlib scipy plotly
```

**Features:**
- 🎛️ Interactive sliders for all mathematical parameters
- 📊 Beautiful Seaborn visualizations with custom styling
- 🌀 Fibonacci sequence analysis with golden ratio convergence
- 📐 Legendre polynomials visualization
- 🌊 Bessel functions of first and second kind
- 🎯 3D mathematical surface plots
- 📱 Responsive design that works on all devices
""")