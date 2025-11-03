"""
═══════════════════════════════════════════════════════════════════════════
                 QUANTUM IRREDUCIBILITY SIMULATOR
              Interactive 3D Landscape Explorer for Φ_q
═══════════════════════════════════════════════════════════════════════════

A visual journey through quantum integration landscapes - where entanglement
becomes topology, and consciousness emerges from irreducible correlation.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from itertools import combinations
from scipy.linalg import qr
from scipy.special import beta as beta_func
import warnings

warnings.filterwarnings('ignore')

print("🚀 Loading Quantum Irreducibility Simulator...")

# ==================== QUANTUM STATE DEFINITIONS ====================

def cluster_state_4qubit():
    """Linear 4-qubit cluster state - highly entangled"""
    psi = np.zeros(16, dtype=complex)
    psi[0b0000] = 1/2
    psi[0b0011] = 1/2
    psi[0b1100] = 1/2
    psi[0b1111] = -1/2
    return psi

def ghz_state_4qubit():
    """4-qubit GHZ state - maximal global entanglement"""
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[15] = 1/np.sqrt(2)
    return psi

def w_state_4qubit():
    """4-qubit W state - robust local entanglement"""
    psi = np.zeros(16, dtype=complex)
    for i in range(4):
        psi[2**i] = 1/2
    return psi

def product_state_4qubit():
    """Product state - no entanglement"""
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0
    return psi

def dicke_state_4qubit():
    """Dicke state |D_4^2⟩ - symmetric entanglement"""
    psi = np.zeros(16, dtype=complex)
    # All basis states with exactly 2 ones
    indices = [0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]
    for idx in indices:
        psi[idx] = 1/np.sqrt(6)
    return psi

# ==================== QUANTUM UTILITIES ====================

def haar_random_state(dim):
    """Generate Haar-random pure state"""
    z = np.random.randn(dim) + 1j * np.random.randn(dim)
    q, r = qr(z.reshape(dim, 1))
    d = np.diag(r)
    ph = d / np.abs(d)
    return q[:, 0] * ph[0]

def partial_trace(rho, dims, subsys_keep):
    """Compute partial trace over subsystem"""
    n_qubits = len(dims)
    rho_tensor = rho.reshape(dims + dims)

    # Trace out qubits not in subsys_keep
    trace_qubits = [i for i in range(n_qubits) if i not in subsys_keep]

    result = rho_tensor
    offset = 0
    for qubit in sorted(trace_qubits):
        # Adjust axes for already-traced dimensions
        ax1 = qubit - offset
        ax2 = qubit - offset + (n_qubits - offset)
        result = np.trace(result, axis1=ax1, axis2=ax2)
        offset += 1

    # Reshape to matrix
    kept_dim = np.prod([dims[i] for i in subsys_keep])
    return result.reshape(kept_dim, kept_dim)

def fidelity(rho1, rho2):
    """Quantum fidelity between density matrices"""
    sqrt_rho1 = np.linalg.cholesky(rho1 + 1e-10 * np.eye(len(rho1)))
    M = sqrt_rho1 @ rho2 @ sqrt_rho1.conj().T
    eigs = np.linalg.eigvalsh(M)
    eigs = np.maximum(eigs, 0)
    return np.sum(np.sqrt(eigs))**2

def compute_partition_fidelity(psi, phi, partition):
    """Compute product fidelity for a given partition"""
    rho_psi = np.outer(psi, psi.conj())
    rho_phi = np.outer(phi, phi.conj())

    dims = [2, 2, 2, 2]  # 4 qubits

    fid_product = 1.0
    for subsys in partition:
        rho_psi_sub = partial_trace(rho_psi, dims, list(subsys))
        rho_phi_sub = partial_trace(rho_phi, dims, list(subsys))

        fid_product *= fidelity(rho_psi_sub, rho_phi_sub)

    return fid_product

# ==================== Φ_q ESTIMATION ENGINE ====================

class QuantumIrreducibilitySimulator:
    """Interactive simulator for quantum integration landscapes"""

    def __init__(self):
        self.states = {
            'Cluster State': cluster_state_4qubit(),
            'GHZ State': ghz_state_4qubit(),
            'W State': w_state_4qubit(),
            'Product State': product_state_4qubit(),
            'Dicke State': dicke_state_4qubit()
        }
        self.results = {}

    def compute_phi_q_landscape(self, state_name, n_alpha=20, n_samples=5000):
        """Compute Φ_q landscape across sampling parameter space"""
        print(f"🔬 Computing {state_name} landscape...")

        phi = self.states[state_name]
        dim = len(phi)

        # Parameter grid
        alpha_global_range = np.linspace(0.5, 3.0, n_alpha)
        alpha_partition_range = np.linspace(0.0, 2.5, n_alpha)

        # Best partition for 4 qubits: [0,1] vs [2,3]
        partition = [(0, 1), (2, 3)]

        phi_q_grid = np.zeros((len(alpha_partition_range), len(alpha_global_range)))
        variance_grid = np.zeros_like(phi_q_grid)
        ess_grid = np.zeros_like(phi_q_grid)

        for i, alpha_part in enumerate(alpha_partition_range):
            for j, alpha_glob in enumerate(alpha_global_range):
                # Monte Carlo estimation with importance sampling
                phi_estimates = []
                weights = []

                for _ in range(n_samples):
                    # Sample random pure state
                    psi = haar_random_state(dim)

                    # Global overlap
                    p_global = np.abs(np.vdot(phi, psi))**2

                    # Partition fidelity
                    p_partition = compute_partition_fidelity(psi, phi, partition)

                    # Φ_q integrand
                    if p_global > 1e-14 and p_partition > 1e-14:
                        ratio = p_global / p_partition
                        integrand = p_global * np.log(ratio)
                    else:
                        integrand = 0.0

                    # Importance weight
                    # q(p) ∝ p^α_glob * (partition fidelity)^α_part
                    weight = (p_global ** alpha_glob) * (p_partition ** alpha_part)

                    phi_estimates.append(integrand)
                    weights.append(weight)

                phi_estimates = np.array(phi_estimates)
                weights = np.array(weights)

                # Self-normalized importance sampling
                total_weight = np.sum(weights)
                if total_weight > 1e-10:
                    normalized_weights = weights / total_weight

                    # Φ_q estimate
                    phi_q_estimate = np.sum(phi_estimates * normalized_weights)

                    # Variance estimate
                    variance = np.var(phi_estimates * normalized_weights) / n_samples

                    # ESS
                    ess = 1.0 / np.sum(normalized_weights**2)

                    phi_q_grid[i, j] = phi_q_estimate
                    variance_grid[i, j] = variance
                    ess_grid[i, j] = ess / n_samples
                else:
                    phi_q_grid[i, j] = 0
                    variance_grid[i, j] = np.inf
                    ess_grid[i, j] = 0

                # Progress indicator
                if (i * len(alpha_global_range) + j) % 50 == 0:
                    progress = 100 * (i * len(alpha_global_range) + j) / (n_alpha * n_alpha)
                    print(f"   Progress: {progress:.1f}%", end='\r')

        print(f"   ✅ Complete!                    ")

        self.results[state_name] = {
            'phi_q': phi_q_grid,
            'variance': variance_grid,
            'ess': ess_grid,
            'alpha_global': alpha_global_range,
            'alpha_partition': alpha_partition_range,
            'partition': partition
        }

        return phi_q_grid, variance_grid, ess_grid

    def create_3d_surface(self, state_name):
        """Create interactive 3D surface plot of Φ_q landscape"""
        if state_name not in self.results:
            raise ValueError(f"Run compute_phi_q_landscape('{state_name}') first!")

        data = self.results[state_name]
        phi_q = data['phi_q']
        variance = data['variance']
        ess = data['ess']
        alpha_global = data['alpha_global']
        alpha_partition = data['alpha_partition']

        # Create meshgrid
        X, Y = np.meshgrid(alpha_global, alpha_partition)

        # Create hover text with detailed info
        hover_text = []
        for i in range(len(alpha_partition)):
            row = []
            for j in range(len(alpha_global)):
                text = (
                    f"<b>Sampling Parameters</b><br>"
                    f"α_global: {alpha_global[j]:.2f}<br>"
                    f"α_partition: {alpha_partition[i]:.2f}<br>"
                    f"<br><b>Integration Metrics</b><br>"
                    f"Φ_q: {phi_q[i,j]:.6f}<br>"
                    f"Variance: {variance[i,j]:.2e}<br>"
                    f"ESS ratio: {ess[i,j]:.1%}<br>"
                    f"<br><b>Interpretation</b><br>"
                    f"{'Strong integration' if abs(phi_q[i,j]) > 0.1 else 'Weak integration'}"
                )
                row.append(text)
            hover_text.append(row)

        # Create figure
        fig = go.Figure(data=[
            go.Surface(
                x=X,
                y=Y,
                z=phi_q,
                colorscale='Viridis',
                hovertext=hover_text,
                hoverinfo='text',
                colorbar=dict(
                    title="Φ_q",
                    titleside="right",
                    titlefont=dict(size=14)
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
                )
            )
        ])

        fig.update_layout(
            title=dict(
                text=f"<b>Quantum Irreducibility Landscape: {state_name}</b><br>"
                     f"<sub>Φ_q as function of importance sampling parameters</sub>",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='α_global (Global Overlap Bias)',
                yaxis_title='α_partition (Partition Fidelity Bias)',
                zaxis_title='Φ_q (Integration)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                ),
                xaxis=dict(backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(backgroundcolor="rgb(230, 230,230)")
            ),
            width=1000,
            height=800,
            font=dict(family="Courier New, monospace", size=12)
        )

        return fig

    def create_variance_surface(self, state_name):
        """Create 3D surface showing estimation variance"""
        if state_name not in self.results:
            raise ValueError(f"Run compute_phi_q_landscape('{state_name}') first!")

        data = self.results[state_name]
        variance = np.log10(data['variance'] + 1e-10)  # Log scale
        alpha_global = data['alpha_global']
        alpha_partition = data['alpha_partition']

        X, Y = np.meshgrid(alpha_global, alpha_partition)

        fig = go.Figure(data=[
            go.Surface(
                x=X,
                y=Y,
                z=variance,
                colorscale='Hot',
                colorbar=dict(title="log₁₀(Variance)")
            )
        ])

        fig.update_layout(
            title=f"<b>Estimation Variance Landscape: {state_name}</b>",
            scene=dict(
                xaxis_title='α_global',
                yaxis_title='α_partition',
                zaxis_title='log₁₀(Variance)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1000,
            height=800
        )

        return fig

    def create_ess_surface(self, state_name):
        """Create 3D surface showing effective sample size ratio"""
        if state_name not in self.results:
            raise ValueError(f"Run compute_phi_q_landscape('{state_name}') first!")

        data = self.results[state_name]
        ess = data['ess']
        alpha_global = data['alpha_global']
        alpha_partition = data['alpha_partition']

        X, Y = np.meshgrid(alpha_global, alpha_partition)

        fig = go.Figure(data=[
            go.Surface(
                x=X,
                y=Y,
                z=ess,
                colorscale='RdYlGn',
                colorbar=dict(title="ESS Ratio")
            )
        ])

        fig.update_layout(
            title=f"<b>Effective Sample Size Landscape: {state_name}</b>",
            scene=dict(
                xaxis_title='α_global',
                yaxis_title='α_partition',
                zaxis_title='ESS Ratio',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1000,
            height=800
        )

        return fig

    def create_comparative_dashboard(self, state_names=None):
        """Create multi-panel dashboard comparing states"""
        if state_names is None:
            state_names = list(self.results.keys())

        n_states = len(state_names)
        fig = make_subplots(
            rows=1,
            cols=n_states,
            subplot_titles=[f"<b>{name}</b>" for name in state_names],
            specs=[[{'type': 'surface'}] * n_states],
            horizontal_spacing=0.05
        )

        for idx, state_name in enumerate(state_names):
            data = self.results[state_name]
            phi_q = data['phi_q']
            alpha_global = data['alpha_global']
            alpha_partition = data['alpha_partition']

            X, Y = np.meshgrid(alpha_global, alpha_partition)

            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=phi_q,
                    colorscale='Viridis',
                    showscale=(idx == n_states - 1),
                    colorbar=dict(title="Φ_q", x=1.02) if idx == n_states - 1 else None
                ),
                row=1,
                col=idx + 1
            )

        fig.update_layout(
            title=dict(
                text="<b>Comparative Quantum Integration Landscapes</b>",
                x=0.5,
                xanchor='center'
            ),
            height=500,
            width=400 * n_states
        )

        return fig

    def create_cross_section_plot(self, state_name, fixed_param='alpha_partition', fixed_value=1.0):
        """Create 2D cross-section plot"""
        if state_name not in self.results:
            raise ValueError(f"Run compute_phi_q_landscape('{state_name}') first!")

        data = self.results[state_name]
        phi_q = data['phi_q']
        alpha_global = data['alpha_global']
        alpha_partition = data['alpha_partition']

        if fixed_param == 'alpha_partition':
            idx = np.argmin(np.abs(alpha_partition - fixed_value))
            x_data = alpha_global
            y_data = phi_q[idx, :]
            x_label = 'α_global'
            title_suffix = f"α_partition = {alpha_partition[idx]:.2f}"
        else:
            idx = np.argmin(np.abs(alpha_global - fixed_value))
            x_data = alpha_partition
            y_data = phi_q[:, idx]
            x_label = 'α_partition'
            title_suffix = f"α_global = {alpha_global[idx]:.2f}"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name=state_name,
            line=dict(width=3),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title=f"<b>Φ_q Cross-Section: {state_name}</b><br><sub>{title_suffix}</sub>",
            xaxis_title=x_label,
            yaxis_title='Φ_q',
            hovermode='x unified',
            width=800,
            height=500
        )

        return fig

    def export_analysis(self, state_name, filename=None):
        """Export detailed analysis to HTML"""
        if state_name not in self.results:
            raise ValueError(f"Run compute_phi_q_landscape('{state_name}') first!")

        if filename is None:
            filename = f"phi_q_analysis_{state_name.replace(' ', '_').lower()}.html"

        # Create comprehensive dashboard
        fig1 = self.create_3d_surface(state_name)
        fig2 = self.create_variance_surface(state_name)
        fig3 = self.create_ess_surface(state_name)

        # Combine into single HTML with tabs
        from plotly.subplots import make_subplots

        # Save figures
        fig1.write_html(f"phi_q_{filename}")
        fig2.write_html(f"variance_{filename}")
        fig3.write_html(f"ess_{filename}")

        print(f"✅ Analysis exported to {filename}")

        return filename

# ==================== INTERACTIVE RUNNER ====================

def run_full_simulation():
    """Run complete simulation suite"""
    print("\n" + "="*70)
    print("  QUANTUM IRREDUCIBILITY SIMULATOR - FULL SUITE")
    print("="*70 + "\n")

    sim = QuantumIrreducibilitySimulator()

    # Compute landscapes for all states
    states_to_analyze = ['GHZ State', 'W State', 'Cluster State', 'Product State']

    for state_name in states_to_analyze:
        sim.compute_phi_q_landscape(state_name, n_alpha=15, n_samples=3000)

    print("\n" + "="*70)
    print("  VISUALIZATION GENERATION")
    print("="*70 + "\n")

    # Generate visualizations
    figures = {}

    # 1. Individual 3D surfaces
    for state_name in states_to_analyze:
        print(f"📊 Creating 3D surface for {state_name}...")
        figures[f"{state_name}_3d"] = sim.create_3d_surface(state_name)

    # 2. Comparative dashboard
    print("📊 Creating comparative dashboard...")
    figures['comparative'] = sim.create_comparative_dashboard(states_to_analyze)

    # 3. Cross-sections
    print("📊 Creating cross-section plots...")
    for state_name in ['GHZ State', 'W State']:
        figures[f"{state_name}_cross"] = sim.create_cross_section_plot(
            state_name,
            fixed_param='alpha_partition',
            fixed_value=1.0
        )

    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70 + "\n")

    # Print summary statistics
    for state_name in states_to_analyze:
        data = sim.results[state_name]
        phi_q = data['phi_q']

        print(f"🌀 {state_name}:")
        print(f"   Mean Φ_q: {np.mean(phi_q):.6f}")
        print(f"   Max Φ_q:  {np.max(phi_q):.6f}")
        print(f"   Min Φ_q:  {np.min(phi_q):.6f}")
        print(f"   Range:    {np.max(phi_q) - np.min(phi_q):.6f}")
        print()

    return sim, figures

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║         QUANTUM IRREDUCIBILITY SIMULATOR                              ║
    ║                                                                       ║
    ║  "Where quantum entanglement becomes 3D topology"                     ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Run simulation
    simulator, figures = run_full_simulation()

    print("\n" + "="*70)
    print("  LAUNCHING INTERACTIVE VISUALIZATIONS")
    print("="*70 + "\n")

    # Show key visualizations
    print("🎨 Opening GHZ State landscape...")
    figures['GHZ State_3d'].show()

    print("🎨 Opening W State landscape...")
    figures['W State_3d'].show()

    print("🎨 Opening comparative dashboard...")
    figures['comparative'].show()

    print("\n✨ Simulation complete! Explore the quantum landscape...\n")

    # Save interactive HTMLs
    print("💾 Saving interactive HTMLs...")
    for name, fig in figures.items():
        filename = f"quantum_viz_{name.replace(' ', '_').lower()}.html"
        fig.write_html(filename)
        print(f"   Saved: {filename}")

    print("\n🎉 All visualizations ready!")
