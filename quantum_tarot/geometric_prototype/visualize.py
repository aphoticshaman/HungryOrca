"""
3D Visualization of Tarot Semantic Space
=========================================

Interactive Plotly visualization showing:
- Card positions as spheres
- Overlap regions
- User query vector
- Reading centroid
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from semantic_space import SemanticSpace, UserProfile, CARD_EMBEDDINGS


def create_3d_space_plot(space: SemanticSpace, selected_cards=None,
                         user_profile: UserProfile = None):
    """
    Create interactive 3D visualization of semantic space.

    Args:
        space: SemanticSpace instance
        selected_cards: List of (card_index, reversed) tuples for active reading
        user_profile: UserProfile to show as query vector
    """
    fig = go.Figure()

    # Plot all available cards as small gray spheres
    all_indices = list(CARD_EMBEDDINGS.keys())
    all_embeddings = np.array([CARD_EMBEDDINGS[idx][1] for idx in all_indices])
    all_names = [CARD_EMBEDDINGS[idx][0] for idx in all_indices]

    fig.add_trace(go.Scatter3d(
        x=all_embeddings[:, 0],
        y=all_embeddings[:, 1],
        z=all_embeddings[:, 2],
        mode='markers+text',
        marker=dict(
            size=8,
            color='lightgray',
            opacity=0.4,
            line=dict(color='gray', width=1)
        ),
        text=all_names,
        textposition='top center',
        textfont=dict(size=8, color='gray'),
        name='All Cards',
        hovertemplate='<b>%{text}</b><br>' +
                     'Elemental: %{x:.2f}<br>' +
                     'Consciousness: %{y:.2f}<br>' +
                     'Temporal: %{z:.2f}<br>' +
                     '<extra></extra>'
    ))

    # If specific cards selected for reading, highlight them
    if selected_cards:
        cards = [space.get_card(idx, reversed) for idx, reversed in selected_cards]
        reading_embeddings = np.array([c.embedding for c in cards])
        reading_names = [c.name for c in cards]
        reading_radii = [c.radius * 50 for c in cards]  # Scale for visibility

        # Plot selected cards as large colored spheres
        fig.add_trace(go.Scatter3d(
            x=reading_embeddings[:, 0],
            y=reading_embeddings[:, 1],
            z=reading_embeddings[:, 2],
            mode='markers+text',
            marker=dict(
                size=reading_radii,
                color=['red', 'blue', 'green'][:len(cards)],
                opacity=0.7,
                line=dict(color='white', width=2)
            ),
            text=reading_names,
            textposition='top center',
            textfont=dict(size=12, color='white'),
            name='Reading Cards',
            hovertemplate='<b>%{text}</b><br>' +
                         'Elemental: %{x:.2f}<br>' +
                         'Consciousness: %{y:.2f}<br>' +
                         'Temporal: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))

        # Draw lines connecting reading cards (show relationships)
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                overlap = space.semantic_similarity(cards[i], cards[j])
                # Only draw line if significant overlap
                if overlap > 0.3:
                    fig.add_trace(go.Scatter3d(
                        x=[reading_embeddings[i, 0], reading_embeddings[j, 0]],
                        y=[reading_embeddings[i, 1], reading_embeddings[j, 1]],
                        z=[reading_embeddings[i, 2], reading_embeddings[j, 2]],
                        mode='lines',
                        line=dict(
                            color='yellow',
                            width=overlap * 10,  # Thicker = stronger overlap
                        ),
                        opacity=0.5,
                        showlegend=False,
                        hovertemplate=f'Overlap: {overlap:.3f}<extra></extra>'
                    ))

        # Plot centroid
        centroid = space.get_centroid(cards)
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2]],
            mode='markers',
            marker=dict(
                size=15,
                color='gold',
                symbol='diamond',
                line=dict(color='orange', width=3)
            ),
            name='Reading Centroid',
            hovertemplate='<b>Reading Center</b><br>' +
                         'Elemental: %{x:.2f}<br>' +
                         'Consciousness: %{y:.2f}<br>' +
                         'Temporal: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))

    # If user profile provided, plot as query vector
    if user_profile:
        fig.add_trace(go.Scatter3d(
            x=[user_profile.vector[0]],
            y=[user_profile.vector[1]],
            z=[user_profile.vector[2]],
            mode='markers+text',
            marker=dict(
                size=20,
                color='purple',
                symbol='square',
                line=dict(color='magenta', width=3)
            ),
            text=[f'User ({user_profile.mbti})'],
            textposition='bottom center',
            textfont=dict(size=14, color='purple'),
            name='User Profile',
            hovertemplate='<b>User Query Vector</b><br>' +
                         f'MBTI: {user_profile.mbti}<br>' +
                         'Elemental: %{x:.2f}<br>' +
                         'Consciousness: %{y:.2f}<br>' +
                         'Temporal: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))

    # Layout configuration
    fig.update_layout(
        title=dict(
            text='Tarot Semantic Space (3D Continuous)',
            font=dict(size=20, color='white')
        ),
        scene=dict(
            xaxis=dict(
                title='Elemental Polarity<br>(Fire-Air ← → Water-Earth)',
                backgroundcolor='rgb(20, 20, 30)',
                gridcolor='rgb(50, 50, 60)',
                showbackground=True,
                range=[-1.2, 1.2]
            ),
            yaxis=dict(
                title='Consciousness Depth<br>(Ego ← → Shadow)',
                backgroundcolor='rgb(20, 20, 30)',
                gridcolor='rgb(50, 50, 60)',
                showbackground=True,
                range=[-1.2, 1.2]
            ),
            zaxis=dict(
                title='Temporal Focus<br>(Past ← → Future)',
                backgroundcolor='rgb(20, 20, 30)',
                gridcolor='rgb(50, 50, 60)',
                showbackground=True,
                range=[-1.2, 1.2]
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        paper_bgcolor='rgb(10, 10, 15)',
        plot_bgcolor='rgb(15, 15, 20)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(30, 30, 40, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        height=800
    )

    return fig


def visualize_overlap_heatmap(space: SemanticSpace, selected_cards):
    """
    Create 2D heatmap of pairwise card overlaps.
    """
    cards = [space.get_card(idx, reversed) for idx, reversed in selected_cards]
    overlaps = space.compute_overlap_strength(cards)
    card_names = [c.name for c in cards]

    fig = go.Figure(data=go.Heatmap(
        z=overlaps,
        x=card_names,
        y=card_names,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='%{y} ↔ %{x}<br>Overlap: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Overlap<br>Strength')
    ))

    fig.update_layout(
        title='Pairwise Card Overlap Matrix',
        xaxis=dict(title='Card', tickangle=-45),
        yaxis=dict(title='Card'),
        paper_bgcolor='rgb(10, 10, 15)',
        plot_bgcolor='rgb(15, 15, 20)',
        font=dict(color='white'),
        height=600
    )

    return fig


if __name__ == "__main__":
    # Initialize space
    space = SemanticSpace()

    # Example reading: Tower (upright) + Death (upright) + Star (upright)
    reading = [
        (16, False),  # The Tower
        (13, False),  # Death
        (17, False),  # The Star
    ]

    # User profile
    user = UserProfile(mbti="INTJ", shadow_integration=0.4, temporal_focus=0.6)

    # Create 3D visualization
    print("Creating 3D semantic space visualization...")
    fig_3d = create_3d_space_plot(space, selected_cards=reading, user_profile=user)
    fig_3d.write_html('semantic_space_3d.html')
    print("✓ Saved to: semantic_space_3d.html")

    # Create overlap heatmap
    print("Creating overlap heatmap...")
    fig_heatmap = visualize_overlap_heatmap(space, reading)
    fig_heatmap.write_html('overlap_heatmap.html')
    print("✓ Saved to: overlap_heatmap.html")

    print("\nVisualization complete! Open the HTML files in a browser to explore.")
    print("\nKey insights:")
    print("- Card size = influence radius")
    print("- Line thickness = overlap strength")
    print("- Gold diamond = reading centroid")
    print("- Purple square = user query vector")
