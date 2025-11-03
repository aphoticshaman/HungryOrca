"""
Quick Quantum Irreducibility Visualization Demo
Fast prototype with reduced sampling for immediate results
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.linalg import qr

print("🚀 Quick Quantum Viz Demo Loading...")

# ==================== QUANTUM STATES ====================

def ghz_state_4qubit():
    """4-qubit GHZ state"""
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[15] = 1/np.sqrt(2)
    return psi

def w_state_4qubit():
    """4-qubit W state"""
    psi = np.zeros(16, dtype=complex)
    for i in range(4):
        psi[2**i] = 1/2
    return psi

def product_state_4qubit():
    """Product state"""
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0
    return psi

def haar_random_state(dim):
    """Haar-random pure state"""
    z = np.random.randn(dim) + 1j * np.random.randn(dim)
    q, r = qr(z.reshape(dim, 1))
    d = np.diag(r)
    ph = d / np.abs(d)
    return q[:, 0] * ph[0]

# ==================== SIMPLE Φ_q ESTIMATOR ====================

def compute_phi_q_simple(phi, alpha_global=1.5, alpha_partition=0.5, n_samples=500):
    """Simplified Φ_q estimation for speed"""
    dim = len(phi)

    phi_estimates = []
    weights = []

    for _ in range(n_samples):
        psi = haar_random_state(dim)
        p_global = np.abs(np.vdot(phi, psi))**2

        # Simplified partition fidelity (just use product of marginals)
        p_partition = (1/4) * (1/4)  # Approximate for 2x2 split

        # Integrand
        if p_global > 1e-14 and p_partition > 1e-14:
            ratio = p_global / p_partition
            integrand = p_global * np.log(ratio)
        else:
            integrand = 0.0

        # Importance weight
        weight = (p_global ** alpha_global) * (p_partition ** alpha_partition)

        phi_estimates.append(integrand)
        weights.append(weight)

    phi_estimates = np.array(phi_estimates)
    weights = np.array(weights)

    total_weight = np.sum(weights)
    if total_weight > 1e-10:
        phi_q = np.sum(phi_estimates * weights) / total_weight
        variance = np.var(phi_estimates * weights / total_weight)
        ess = 1.0 / np.sum((weights / total_weight)**2) / n_samples
    else:
        phi_q = 0
        variance = np.inf
        ess = 0

    return phi_q, variance, ess

# ==================== LANDSCAPE COMPUTATION ====================

def compute_landscape_fast(phi, n_alpha=10, n_samples=500):
    """Fast landscape computation"""
    alpha_global_range = np.linspace(0.5, 3.0, n_alpha)
    alpha_partition_range = np.linspace(0.0, 2.5, n_alpha)

    phi_q_grid = np.zeros((n_alpha, n_alpha))
    variance_grid = np.zeros((n_alpha, n_alpha))
    ess_grid = np.zeros((n_alpha, n_alpha))

    total_iterations = n_alpha * n_alpha
    count = 0

    for i, alpha_part in enumerate(alpha_partition_range):
        for j, alpha_glob in enumerate(alpha_global_range):
            phi_q, var, ess = compute_phi_q_simple(
                phi, alpha_glob, alpha_part, n_samples
            )

            phi_q_grid[i, j] = phi_q
            variance_grid[i, j] = var
            ess_grid[i, j] = ess

            count += 1
            if count % 10 == 0:
                print(f"   Progress: {100*count/total_iterations:.0f}%", end='\r')

    print(f"   ✅ Complete!{' '*20}")

    return alpha_global_range, alpha_partition_range, phi_q_grid, variance_grid, ess_grid

# ==================== 3D VISUALIZATION ====================

def create_3d_interactive(state_name, alpha_global, alpha_partition, phi_q, variance, ess):
    """Create stunning 3D interactive visualization"""

    X, Y = np.meshgrid(alpha_global, alpha_partition)

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Φ_q Integration Landscape', 'log₁₀(Variance)', 'ESS Efficiency'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.05
    )

    # Φ_q surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=phi_q,
            colorscale='RdBu_r',
            name='Φ_q',
            showscale=True,
            colorbar=dict(title="Φ_q", x=0.29),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="lime", project=dict(z=True))
            )
        ),
        row=1, col=1
    )

    # Variance surface
    log_var = np.log10(variance + 1e-10)
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=log_var,
            colorscale='Hot',
            name='Variance',
            showscale=True,
            colorbar=dict(title="log₁₀(Var)", x=0.63)
        ),
        row=1, col=2
    )

    # ESS surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=ess,
            colorscale='Viridis',
            name='ESS',
            showscale=True,
            colorbar=dict(title="ESS Ratio", x=0.97)
        ),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Quantum Irreducibility Landscape: {state_name}</b><br>"
                 f"<sub>Interactive 3D explorer - rotate, zoom, and hover for details</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='α_global',
            yaxis_title='α_partition',
            zaxis_title='Φ_q',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        scene2=dict(
            xaxis_title='α_global',
            yaxis_title='α_partition',
            zaxis_title='log₁₀(Var)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        scene3=dict(
            xaxis_title='α_global',
            yaxis_title='α_partition',
            zaxis_title='ESS Ratio',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1800,
        height=600,
        font=dict(family="Courier New, monospace", size=11)
    )

    return fig

def create_comparative_view(states_data):
    """Create side-by-side comparison of multiple states"""
    n_states = len(states_data)

    fig = make_subplots(
        rows=1, cols=n_states,
        subplot_titles=[f"<b>{name}</b>" for name in states_data.keys()],
        specs=[[{'type': 'surface'}] * n_states],
        horizontal_spacing=0.03
    )

    for idx, (state_name, data) in enumerate(states_data.items()):
        alpha_global, alpha_partition, phi_q, _, _ = data
        X, Y = np.meshgrid(alpha_global, alpha_partition)

        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=phi_q,
                colorscale='Plasma',
                showscale=(idx == n_states - 1),
                colorbar=dict(title="Φ_q", x=1.0) if idx == n_states - 1 else None,
                name=state_name
            ),
            row=1, col=idx + 1
        )

    fig.update_layout(
        title=dict(
            text="<b>Comparative Quantum Integration: GHZ vs W vs Product</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=500,
        width=450 * n_states,
        font=dict(size=10)
    )

    # Update camera for all scenes
    camera_dict = dict(eye=dict(x=1.3, y=1.3, z=1.1))
    for i in range(n_states):
        scene_name = 'scene' if i == 0 else f'scene{i+1}'
        fig.layout[scene_name].camera = camera_dict

    return fig

# ==================== MAIN DEMO ====================

def run_demo():
    """Run complete demonstration"""
    print("\n" + "="*70)
    print("  QUANTUM IRREDUCIBILITY LANDSCAPE - QUICK DEMO")
    print("="*70 + "\n")

    states = {
        'GHZ State': ghz_state_4qubit(),
        'W State': w_state_4qubit(),
        'Product State': product_state_4qubit()
    }

    results = {}

    for state_name, phi in states.items():
        print(f"🔬 Computing {state_name}...")
        landscape_data = compute_landscape_fast(phi, n_alpha=12, n_samples=400)
        results[state_name] = landscape_data

        alpha_global, alpha_partition, phi_q, variance, ess = landscape_data

        print(f"   Mean Φ_q: {np.mean(phi_q):.6f}")
        print(f"   Max Φ_q:  {np.max(phi_q):.6f}")
        print(f"   Min Φ_q:  {np.min(phi_q):.6f}\n")

    print("="*70)
    print("  GENERATING VISUALIZATIONS")
    print("="*70 + "\n")

    # Create individual detailed views
    figures = {}
    for state_name, data in results.items():
        print(f"📊 Creating 3D landscape for {state_name}...")
        fig = create_3d_interactive(state_name, *data)
        figures[state_name] = fig

        # Save HTML
        filename = f"quantum_landscape_{state_name.replace(' ', '_').lower()}.html"
        fig.write_html(filename)
        print(f"   ✅ Saved: {filename}")

    # Create comparative view
    print("\n📊 Creating comparative dashboard...")
    comp_fig = create_comparative_view(results)
    comp_fig.write_html("quantum_landscape_comparison.html")
    print("   ✅ Saved: quantum_landscape_comparison.html")

    print("\n" + "="*70)
    print("  LAUNCHING INTERACTIVE VISUALIZATIONS")
    print("="*70 + "\n")

    # Display figures
    print("🎨 Opening GHZ State landscape...")
    figures['GHZ State'].show()

    print("🎨 Opening comparison view...")
    comp_fig.show()

    print("\n✨ Demo complete! HTML files saved for offline exploration.\n")

    return figures, results

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║      QUANTUM IRREDUCIBILITY LANDSCAPE - QUICK DEMO                   ║
    ║                                                                       ║
    ║  "Exploring the topology of quantum entanglement in 3D"              ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    figures, results = run_demo()

    print("\n🎉 All visualizations ready for exploration!")
    print("   Open the HTML files in your browser for full interactivity.")
    print("   Use mouse to rotate, scroll to zoom, hover for details.\n")
