"""
═══════════════════════════════════════════════════════════════════════════
                    ARC GRID Φ_q IRREDUCIBILITY EXPLORER
              Apply Quantum Integration Metrics to Pattern Transformations
═══════════════════════════════════════════════════════════════════════════

Analyzes how transformations (rotation, reflection, etc.) affect pattern
irreducibility - measuring whether the output is "more whole" than the input.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("🎨 Loading ARC Grid Φ_q Explorer...")

# ═══════════════════════════════════════════════════════════════════════════
# GRID TO QUANTUM STATE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

def grid_to_statevector(grid):
    """
    Convert ARC grid to quantum-like statevector

    Flattens grid, normalizes to unit vector, pads to power of 2
    """
    vec = grid.flatten().astype(complex)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    else:
        vec = np.ones(len(vec), dtype=complex) / np.sqrt(len(vec))

    # Pad to next power of 2 for quantum compatibility
    n_qubits = int(np.ceil(np.log2(len(vec))))
    next_pow2 = 2 ** n_qubits
    if next_pow2 > len(vec):
        vec = np.pad(vec, (0, next_pow2 - len(vec)))

    return vec

def haar_random_state(dim):
    """Generate Haar-random pure state (probe)"""
    z = np.random.randn(dim) + 1j * np.random.randn(dim)
    norm = np.linalg.norm(z)
    if norm > 0:
        z /= norm
    return z

# ═══════════════════════════════════════════════════════════════════════════
# SUBSYSTEM PARTITIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_row_partition(grid_shape):
    """Partition grid into rows as subsystems"""
    rows, cols = grid_shape
    subsystems = []
    for r in range(rows):
        subsystems.append(list(range(r * cols, (r + 1) * cols)))
    return subsystems

def get_column_partition(grid_shape):
    """Partition grid into columns as subsystems"""
    rows, cols = grid_shape
    subsystems = []
    for c in range(cols):
        subsystems.append(list(range(c, rows * cols, cols)))
    return subsystems

def get_quadrant_partition(grid_shape):
    """Partition grid into quadrants (for even-sized grids)"""
    rows, cols = grid_shape
    if rows % 2 != 0 or cols % 2 != 0:
        # Fall back to row partition for odd sizes
        return get_row_partition(grid_shape)

    half_r, half_c = rows // 2, cols // 2

    # Top-left quadrant
    q1 = []
    for r in range(half_r):
        for c in range(half_c):
            q1.append(r * cols + c)

    # Top-right quadrant
    q2 = []
    for r in range(half_r):
        for c in range(half_c, cols):
            q2.append(r * cols + c)

    # Bottom-left quadrant
    q3 = []
    for r in range(half_r, rows):
        for c in range(half_c):
            q3.append(r * cols + c)

    # Bottom-right quadrant
    q4 = []
    for r in range(half_r, rows):
        for c in range(half_c, cols):
            q4.append(r * cols + c)

    return [q1, q2, q3, q4]

# ═══════════════════════════════════════════════════════════════════════════
# PARTIAL FIDELITY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def partial_fidelity_subsystem(psi, phi, subsys):
    """
    Compute fidelity between probe and target on subsystem

    F = |⟨ψ_sub|φ_sub⟩|²
    """
    # Extract subsystem components
    psi_sub = psi[subsys]
    phi_sub = phi[subsys]

    # Normalize
    norm_psi = np.linalg.norm(psi_sub)
    norm_phi = np.linalg.norm(phi_sub)

    if norm_psi > 0:
        psi_sub = psi_sub / norm_psi
    if norm_phi > 0:
        phi_sub = phi_sub / norm_phi

    # Fidelity = squared overlap
    fidelity = np.abs(np.vdot(phi_sub, psi_sub)) ** 2

    return fidelity

# ═══════════════════════════════════════════════════════════════════════════
# Φ_q COMPUTATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def compute_phi_q_single(phi, subsystems, n_samples=500, alpha_global=1.0, alpha_sub=0.5):
    """
    Compute Φ_q for a single (α_global, α_sub) parameter pair

    Returns: Φ_q estimate, variance, ESS ratio
    """
    estimates = []
    weights = []

    for _ in range(n_samples):
        # Generate random probe state
        psi = haar_random_state(len(phi))

        # Global overlap
        p_global = np.abs(np.vdot(phi, psi)) ** 2

        # Subsystem fidelities
        p_subs = [partial_fidelity_subsystem(psi, phi, s) for s in subsystems]

        # Importance weight: p^α_g × ∏(p_sub^α_s)
        w = (p_global ** alpha_global) * np.prod([p ** alpha_sub for p in p_subs])

        # Integrand: p_global × log(p_global / mean(p_subs))
        mean_sub = max(np.mean(p_subs), 1e-12)
        integrand = p_global * np.log(max(p_global / mean_sub, 1e-12))

        estimates.append(integrand)
        weights.append(w)

    estimates = np.array(estimates)
    weights = np.array(weights)

    # Self-normalized importance sampling
    total_weight = np.sum(weights)
    if total_weight > 1e-12:
        normalized_weights = weights / total_weight

        # Φ_q estimate
        phi_q_estimate = np.sum(estimates * normalized_weights)

        # Variance
        variance = np.var(estimates * normalized_weights)

        # ESS
        ess = 1.0 / np.sum(normalized_weights ** 2)
        ess_ratio = ess / n_samples
    else:
        phi_q_estimate = 0.0
        variance = np.inf
        ess_ratio = 0.0

    return phi_q_estimate, variance, ess_ratio

def compute_phi_q_landscape(phi, subsystems, n_alpha=12, n_samples=500):
    """
    Compute full 2D landscape over (α_global, α_sub) parameter space

    Returns: alpha_global_vals, alpha_sub_vals, phi_q_grid, variance_grid, ess_grid
    """
    alpha_global_vals = np.linspace(0.5, 3.0, n_alpha)
    alpha_sub_vals = np.linspace(0.0, 2.5, n_alpha)

    phi_q_grid = np.zeros((len(alpha_sub_vals), len(alpha_global_vals)))
    variance_grid = np.zeros_like(phi_q_grid)
    ess_grid = np.zeros_like(phi_q_grid)

    print(f"Computing {n_alpha}×{n_alpha} landscape...")

    for i, alpha_sub in enumerate(alpha_sub_vals):
        for j, alpha_global in enumerate(alpha_global_vals):
            phi_q, var, ess = compute_phi_q_single(
                phi, subsystems, n_samples=n_samples,
                alpha_global=alpha_global, alpha_sub=alpha_sub
            )

            phi_q_grid[i, j] = phi_q
            variance_grid[i, j] = var
            ess_grid[i, j] = ess

        print(f"   Progress: {i+1}/{len(alpha_sub_vals)} rows", end='\r')

    print(f"   ✅ Complete!{' '*30}")

    return alpha_global_vals, alpha_sub_vals, phi_q_grid, variance_grid, ess_grid

# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════════════════

def rotate_90_cw(grid):
    """Rotate grid 90° clockwise"""
    return np.rot90(grid, k=-1)

def rotate_90_ccw(grid):
    """Rotate grid 90° counter-clockwise"""
    return np.rot90(grid, k=1)

def rotate_180(grid):
    """Rotate grid 180°"""
    return np.rot90(grid, k=2)

def reflect_horizontal(grid):
    """Reflect grid horizontally (left-right flip)"""
    return np.fliplr(grid)

def reflect_vertical(grid):
    """Reflect grid vertically (up-down flip)"""
    return np.flipud(grid)

def transpose_grid(grid):
    """Transpose grid (reflect across main diagonal)"""
    return grid.T

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def create_3d_landscape(alpha_global, alpha_sub, phi_q, variance, ess, title="Φ_q Landscape"):
    """
    Create interactive 3D Plotly surface with 3 panels:
    - Φ_q integration landscape
    - log₁₀(Variance)
    - ESS efficiency
    """
    X, Y = np.meshgrid(alpha_global, alpha_sub)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            '<b>Φ_q Integration</b>',
            '<b>log₁₀(Variance)</b>',
            '<b>ESS Efficiency</b>'
        ),
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.05
    )

    # Panel 1: Φ_q surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=phi_q,
            colorscale='RdBu',
            name='Φ_q',
            showscale=True,
            colorbar=dict(title="Φ_q", x=0.29),
            hovertemplate='α_global: %{x:.2f}<br>α_sub: %{y:.2f}<br>Φ_q: %{z:.6f}<extra></extra>',
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="lime", project=dict(z=True))
            )
        ),
        row=1, col=1
    )

    # Panel 2: Variance surface (log scale)
    log_var = np.log10(variance + 1e-10)
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=log_var,
            colorscale='Hot',
            name='Variance',
            showscale=True,
            colorbar=dict(title="log₁₀(Var)", x=0.63),
            hovertemplate='α_global: %{x:.2f}<br>α_sub: %{y:.2f}<br>log₁₀(Var): %{z:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Panel 3: ESS surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=ess,
            colorscale='Viridis',
            name='ESS',
            showscale=True,
            colorbar=dict(title="ESS Ratio", x=0.97),
            hovertemplate='α_global: %{x:.2f}<br>α_sub: %{y:.2f}<br>ESS: %{z:.1%}<extra></extra>'
        ),
        row=1, col=3
    )

    # Update layout
    camera_dict = dict(eye=dict(x=1.3, y=1.3, z=1.1))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sub>Interactive 3D Quantum Irreducibility Explorer</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='α_global',
            yaxis_title='α_sub',
            zaxis_title='Φ_q',
            camera=camera_dict
        ),
        scene2=dict(
            xaxis_title='α_global',
            yaxis_title='α_sub',
            zaxis_title='log₁₀(Var)',
            camera=camera_dict
        ),
        scene3=dict(
            xaxis_title='α_global',
            yaxis_title='α_sub',
            zaxis_title='ESS',
            camera=camera_dict
        ),
        width=1800,
        height=600,
        font=dict(family="Courier New, monospace", size=11)
    )

    return fig

def create_comparison_plot(grids_dict, partition_type='rows', n_alpha=12, n_samples=500):
    """
    Compare Φ_q landscapes for multiple grids side-by-side

    Args:
        grids_dict: {name: grid_array}
        partition_type: 'rows', 'columns', or 'quadrants'
    """
    results = {}

    for name, grid in grids_dict.items():
        print(f"\n🔬 Analyzing: {name}")

        # Convert to statevector
        phi = grid_to_statevector(grid)

        # Get subsystems
        if partition_type == 'rows':
            subsystems = get_row_partition(grid.shape)
        elif partition_type == 'columns':
            subsystems = get_column_partition(grid.shape)
        else:
            subsystems = get_quadrant_partition(grid.shape)

        # Compute landscape
        alpha_global, alpha_sub, phi_q, variance, ess = compute_phi_q_landscape(
            phi, subsystems, n_alpha=n_alpha, n_samples=n_samples
        )

        results[name] = {
            'alpha_global': alpha_global,
            'alpha_sub': alpha_sub,
            'phi_q': phi_q,
            'variance': variance,
            'ess': ess,
            'grid': grid
        }

        print(f"   Mean Φ_q: {np.mean(phi_q):.6f}")
        print(f"   Max Φ_q:  {np.max(phi_q):.6f}")
        print(f"   Min Φ_q:  {np.min(phi_q):.6f}")

    # Create side-by-side comparison
    n_grids = len(grids_dict)

    fig = make_subplots(
        rows=1, cols=n_grids,
        subplot_titles=[f"<b>{name}</b>" for name in grids_dict.keys()],
        specs=[[{'type': 'surface'}] * n_grids],
        horizontal_spacing=0.03
    )

    for idx, (name, data) in enumerate(results.items()):
        X, Y = np.meshgrid(data['alpha_global'], data['alpha_sub'])

        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=data['phi_q'],
                colorscale='Plasma',
                showscale=(idx == n_grids - 1),
                colorbar=dict(title="Φ_q", x=1.0) if idx == n_grids - 1 else None,
                name=name,
                hovertemplate=f'{name}<br>α_global: %{{x:.2f}}<br>α_sub: %{{y:.2f}}<br>Φ_q: %{{z:.6f}}<extra></extra>'
            ),
            row=1, col=idx + 1
        )

    # Update layout
    camera_dict = dict(eye=dict(x=1.3, y=1.3, z=1.1))
    for i in range(1, n_grids + 1):
        scene_name = 'scene' if i == 1 else f'scene{i}'
        fig.layout[scene_name].camera = camera_dict
        fig.layout[scene_name].xaxis.title = 'α_global'
        fig.layout[scene_name].yaxis.title = 'α_sub'
        fig.layout[scene_name].zaxis.title = 'Φ_q'

    fig.update_layout(
        title=dict(
            text=f"<b>Transformation Φ_q Comparison ({partition_type} partition)</b>",
            x=0.5,
            xanchor='center'
        ),
        height=500,
        width=500 * n_grids
    )

    return fig, results

# ═══════════════════════════════════════════════════════════════════════════
# DEMO: ROTATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def rotation_analysis_demo():
    """
    Analyze how 90° rotation affects pattern irreducibility
    """
    print("\n" + "="*70)
    print("  ARC GRID Φ_q ROTATION ANALYSIS")
    print("="*70 + "\n")

    # Example 3×3 grid with asymmetric pattern
    original_grid = np.array([
        [0, 1, 0],
        [0, 0, 2],
        [3, 0, 0]
    ], dtype=float)

    # Apply 90° clockwise rotation
    rotated_grid = rotate_90_cw(original_grid)

    print("Original Grid:")
    print(original_grid)
    print("\nRotated Grid (90° CW):")
    print(rotated_grid)

    # Compare landscapes
    grids_dict = {
        'Original': original_grid,
        'Rotated 90°': rotated_grid
    }

    fig, results = create_comparison_plot(
        grids_dict,
        partition_type='rows',
        n_alpha=12,
        n_samples=500
    )

    # Save and show
    filename = "arc_rotation_phi_q_comparison.html"
    fig.write_html(filename)
    print(f"\n✅ Saved: {filename}")

    fig.show()

    # Analysis summary
    print("\n" + "="*70)
    print("  TRANSFORMATION ANALYSIS")
    print("="*70 + "\n")

    orig_phi_q = results['Original']['phi_q']
    rot_phi_q = results['Rotated 90°']['phi_q']

    delta_mean = np.mean(rot_phi_q) - np.mean(orig_phi_q)
    delta_max = np.max(rot_phi_q) - np.max(orig_phi_q)

    print(f"Original → Rotated Change:")
    print(f"   Δ Mean Φ_q: {delta_mean:+.6f}")
    print(f"   Δ Max Φ_q:  {delta_max:+.6f}")

    if delta_mean > 0.01:
        print("\n✨ Rotation INCREASED irreducibility (created more binding)")
    elif delta_mean < -0.01:
        print("\n⚠️  Rotation DECREASED irreducibility (broke some structure)")
    else:
        print("\n➡️  Rotation preserved irreducibility (neutral transformation)")

    return fig, results

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║         ARC GRID Φ_q IRREDUCIBILITY EXPLORER                         ║
    ║                                                                       ║
    ║  "Measuring how transformations affect pattern wholeness"            ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Run rotation analysis demo
    fig, results = rotation_analysis_demo()

    print("\n🎉 Analysis complete! Explore the interactive 3D landscapes.\n")
