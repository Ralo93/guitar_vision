import networkx as nx
import plotly.graph_objects as go

# Step 1: Create a graph structure to represent chords and harmonics
G = nx.Graph()

# Define chords for the Circle of Fifths (Major and Minor)
circle_of_fifths_major = ["C Major", "G Major", "D Major", "A Major", "E Major", "B Major", "F# Major", "Db Major", "Ab Major", "Eb Major", "Bb Major", "F Major"]
circle_of_fifths_minor = ["A Minor", "E Minor", "B Minor", "F# Minor", "C# Minor", "G# Minor", "D# Minor", "Bb Minor", "F Minor", "C Minor", "G Minor", "D Minor"]

# Define advanced chords
augmented_chords = {
    "C Major": "C augmented",
    "G Major": "G augmented",
    "D Major": "D augmented",
    "A Major": "A augmented",
    "E Major": "E augmented",
    "B Major": "B augmented",
    "F# Major": "F# augmented",
    "Db Major": "Db augmented",
    "Ab Major": "Ab augmented",
    "Eb Major": "Eb augmented",
    "Bb Major": "Bb augmented",
    "F Major": "F augmented"
}

dominant_7th_chords = {
    "C Major": "C7",
    "G Major": "G7",
    "D Major": "D7",
    "A Major": "A7",
    "E Major": "E7",
    "B Major": "B7",
    "F# Major": "F#7",
    "Db Major": "Db7",
    "Ab Major": "Ab7",
    "Eb Major": "Eb7",
    "Bb Major": "Bb7",
    "F Major": "F7"
}

major_7th_chords = {
    "C Major": "Cmaj7",
    "G Major": "Gmaj7",
    "D Major": "Dmaj7",
    "A Major": "Amaj7",
    "E Major": "Emaj7",
    "B Major": "Bmaj7",
    "F# Major": "F#maj7",
    "Db Major": "Dbmaj7",
    "Ab Major": "Abmaj7",
    "Eb Major": "Ebmaj7",
    "Bb Major": "Bbmaj7",
    "F Major": "Fmaj7"
}

minor_7th_chords = {
    "A Minor": "Am7",
    "E Minor": "Em7",
    "B Minor": "Bm7",
    "F# Minor": "F#m7",
    "C# Minor": "C#m7",
    "G# Minor": "G#m7",
    "D# Minor": "D#m7",
    "Bb Minor": "Bbm7",
    "F Minor": "Fm7",
    "C Minor": "Cm7",
    "G Minor": "Gm7",
    "D Minor": "Dm7"
}

half_diminished_7th_chords = {
    "B Minor": "Bm7b5",
    "F# Minor": "F#m7b5",
    "C# Minor": "C#m7b5",
    "G# Minor": "G#m7b5",
    "D# Minor": "D#m7b5",
    "Bb Minor": "Bbm7b5",
    "F Minor": "Fm7b5",
    "C Minor": "Cm7b5",
    "G Minor": "Gm7b5",
    "D Minor": "Dm7b5",
    "A Minor": "Am7b5",
    "E Minor": "Em7b5"
}

full_diminished_7th_chords = {
    "B Minor": "Bdim7",
    "F# Minor": "F#dim7",
    "C# Minor": "C#dim7",
    "G# Minor": "G#dim7",
    "D# Minor": "D#dim7",
    "Bb Minor": "Bbdim7",
    "F Minor": "Fdim7",
    "C Minor": "Cdim7",
    "G Minor": "Gdim7",
    "D Minor": "Ddim7",
    "A Minor": "Adim7",
    "E Minor": "Edim7"
}

# Step 2: Add chords and harmonic relationships to the graph
# Add Circle of Fifths (Major)
for i in range(len(circle_of_fifths_major)):
    G.add_edge(circle_of_fifths_major[i], circle_of_fifths_major[(i + 1) % len(circle_of_fifths_major)], label="Circle: Transition through the fifth")

# Add Circle of Fifths (Minor)
for i in range(len(circle_of_fifths_minor)):
    G.add_edge(circle_of_fifths_minor[i], circle_of_fifths_minor[(i + 1) % len(circle_of_fifths_minor)], label="Circle: Transition through the fifth")

# Add advanced chords
for chord, aug in augmented_chords.items():
    G.add_edge(chord, aug, label="Augmented: Raise the fifth degree")

for chord, dom7 in dominant_7th_chords.items():
    G.add_edge(chord, dom7, label="Dominant 7th: Add a minor seventh")

for chord, maj7 in major_7th_chords.items():
    G.add_edge(chord, maj7, label="Major 7th: Add a major seventh")

for chord, min7 in minor_7th_chords.items():
    G.add_edge(chord, min7, label="Minor 7th: Add a minor seventh")

for chord, half_dim7 in half_diminished_7th_chords.items():
    G.add_edge(chord, half_dim7, label="Half-diminished 7th: Minor seventh with diminished fifth")

for chord, full_dim7 in full_diminished_7th_chords.items():
    G.add_edge(chord, full_dim7, label="Full-diminished 7th: Diminished seventh")

# Step 3: Get 3D positions for each node using a spring layout with larger k value to spread the graph
pos = nx.spring_layout(G, dim=3, k=1.5)  # k controls spacing between nodes

# Extract node positions for 3D visualization
x_nodes = [pos[node][0] for node in G.nodes()]
y_nodes = [pos[node][1] for node in G.nodes()]
z_nodes = [pos[node][2] for node in G.nodes()]

# Extract edge positions and labels for 3D visualization
x_edges = []
y_edges = []
z_edges = []
edge_labels = []
for edge in G.edges(data=True):
    x_edges += [pos[edge[0]][0], pos[edge[1]][0], None]
    y_edges += [pos[edge[0]][1], pos[edge[1]][1], None]
    z_edges += [pos[edge[0]][2], pos[edge[1]][2], None]
    edge_labels.append(edge[2]['label'])

# Step 4: Create the 3D plot with plotly
edge_trace = go.Scatter3d(
    x=x_edges, y=y_edges, z=z_edges,
    mode='lines',
    line=dict(color='gray', width=2),
    hoverinfo='none'
)

# Visualizing the node labels and markers
node_trace = go.Scatter3d(
    x=x_nodes, y=y_nodes, z=z_nodes,
    mode='markers+text',
    marker=dict(symbol='circle', size=10, color='blue'),
    text=list(G.nodes()),  # Display chord names as text
    hoverinfo='text'
)

# Step 5: Add labels for the edges
edge_label_trace = go.Scatter3d(
    x=[(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in G.edges()],
    y=[(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in G.edges()],
    z=[(pos[edge[0]][2] + pos[edge[1]][2]) / 2 for edge in G.edges()],
    mode='text',
    text=edge_labels,  # Display
    textposition="middle center",
    hoverinfo='none'
)
# Define which nodes and edges are relevant for "Easy Blues" in A major
easy_blues_nodes = ["A Major", "D Major", "E Major", "A7", "D7", "E7"]
easy_blues_edges = [
    ("A Major", "A7"),
    ("D Major", "D7"),
    ("E Major", "E7"),
    ("A Major", "D Major"),
    ("D Major", "E Major")
]

# Add a button to toggle "Easy Blues" mode on and off
updatemenus = [
    dict(
        type="buttons",
        direction="left",
        buttons=[
            dict(
                label="Easy Blues",
                method="update",
                args=[
                    {"visible": [False if node not in easy_blues_nodes else True for node in G.nodes()] + 
                                [False if edge not in easy_blues_edges else True for edge in G.edges()]}  # Show only relevant edges and nodes
                ]
            ),
            dict(
                label="Show All",
                method="update",
                args=[
                    {"visible": [True] * len(G.nodes()) + [True] * len(G.edges())}  # Restore all edges and nodes
                ]
            )
        ],
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.1,
        xanchor="left",
        y=1.1,
        yanchor="top"
    )
]


# Step 6: Create and display the 3D figure
fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
fig.update_layout(
    title="3D Visualization of Harmonics - Circle of Fifths, Advanced Chords, and Musical Transitions",
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False),
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    showlegend=False,
    updatemenus=updatemenus  # Add the button here
)
fig.show()
