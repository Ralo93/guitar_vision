import networkx as nx
import plotly.graph_objects as go
import base64

# Step 1: Create a graph structure to represent chords and harmonics
G = nx.Graph()


def create_chord_diagram_svg(chord_name):
    # Basic SVG template for a guitar chord diagram
    svg_template = '''
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="100" viewBox="0 0 80 100">
        <!-- Fretboard -->
        <rect x="10" y="10" width="60" height="80" fill="white" stroke="black"/>
        
        <!-- Vertical lines (strings) -->
        <line x1="20" y1="10" x2="20" y2="90" stroke="black"/>
        <line x1="30" y1="10" x2="30" y2="90" stroke="black"/>
        <line x1="40" y1="10" x2="40" y2="90" stroke="black"/>
        <line x1="50" y1="10" x2="50" y2="90" stroke="black"/>
        <line x1="60" y1="10" x2="60" y2="90" stroke="black"/>
        
        <!-- Horizontal lines (frets) -->
        <line x1="10" y1="30" x2="70" y2="30" stroke="black"/>
        <line x1="10" y1="50" x2="70" y2="50" stroke="black"/>
        <line x1="10" y1="70" x2="70" y2="70" stroke="black"/>
        
        <!-- Chord name -->
        <text x="40" y="100" text-anchor="middle" font-size="12">{}</text>
        
        {dots}
    </svg>
    '''
    
    # Define chord fingerings (simplified for example)
    chord_dots = {
        "Db Major": '''
            <!-- X markers -->
            <text x="20" y="8" text-anchor="middle" font-size="12">×</text>
            <text x="30" y="8" text-anchor="middle" font-size="12">×</text>
            <!-- Finger positions -->
            <circle cx="40" cy="30" r="5" fill="blue"/>
            <circle cx="50" cy="30" r="5" fill="blue"/>
            <circle cx="50" cy="50" r="5" fill="blue"/>
            <circle cx="40" cy="70" r="5" fill="blue"/>
        ''',
        # Add more chord diagrams as needed...
    }
    
    # Get the dots for the specific chord, or use empty string if not defined
    dots = chord_dots.get(chord_name, "")
    
    # Create the complete SVG
    svg = svg_template.format(chord_name, dots=dots)
    
    # Convert SVG to base64 for Plotly
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

# Define chords for the Circle of Fifths (Major and Minor)
circle_of_fifths_major = ["C Major", "G Major", "D Major", "A Major", "E Major", "B Major", "F# Major", "Db Major", "Ab Major", "Eb Major", "Bb Major", "F Major"]
circle_of_fifths_minor = ["A Minor", "E Minor", "B Minor", "F# Minor", "C# Minor", "G# Minor", "D# Minor", "Bb Minor", "F Minor", "C Minor", "G Minor", "D Minor"]

# Define relative major/minor relationships
relative_major_minor = {
    "C Major": ("A Minor", "Down a minor third"),
    "G Major": ("E Minor", "Down a minor third"),
    "D Major": ("B Minor", "Down a minor third"),
    "A Major": ("F# Minor", "Down a minor third"),
    "E Major": ("C# Minor", "Down a minor third"),
    "B Major": ("G# Minor", "Down a minor third"),
    "F# Major": ("D# Minor", "Down a minor third"),
    "Db Major": ("Bb Minor", "Down a minor third"),
    "Ab Major": ("F Minor", "Down a minor third"),
    "Eb Major": ("C Minor", "Down a minor third"),
    "Bb Major": ("G Minor", "Down a minor third"),
    "F Major": ("D Minor", "Down a minor third")
}

# [Previous chord definitions remain the same - suspended_chords, diminished_chords, I_IV_V_major, I_IV_V_minor]
suspended_chords = {
    "C Major": ("C sus2", "C sus4"),
    "G Major": ("G sus2", "G sus4"),
    "D Major": ("D sus2", "D sus4"),
    "A Major": ("A sus2", "A sus4"),
    "E Major": ("E sus2", "E sus4"),
    "B Major": ("B sus2", "B sus4"),
    "F# Major": ("F# sus2", "F# sus4"),
    "Db Major": ("Db sus2", "Db sus4"),
    "Ab Major": ("Ab sus2", "Ab sus4"),
    "Eb Major": ("Eb sus2", "Eb sus4"),
    "Bb Major": ("Bb sus2", "Bb sus4"),
    "F Major": ("F sus2", "F sus4")
}

diminished_chords = {
    "C Major": "C dim",
    "G Major": "G dim",
    "D Major": "D dim",
    "A Major": "A dim",
    "E Major": "E dim",
    "B Major": "B dim",
    "F# Major": "F# dim",
    "Db Major": "Db dim",
    "Ab Major": "Ab dim",
    "Eb Major": "Eb dim",
    "Bb Major": "Bb dim",
    "F Major": "F dim"
}

I_IV_V_major = {
    "C Major": ("F Major", "G Major"),
    "G Major": ("C Major", "D Major"),
    "D Major": ("G Major", "A Major"),
    "A Major": ("D Major", "E Major"),
    "E Major": ("A Major", "B Major"),
    "B Major": ("E Major", "F# Major"),
    "F# Major": ("B Major", "Db Major"),
    "Db Major": ("F# Major", "Ab Major"),
    "Ab Major": ("Db Major", "Eb Major"),
    "Eb Major": ("Ab Major", "Bb Major"),
    "Bb Major": ("Eb Major", "F Major"),
    "F Major": ("Bb Major", "C Major")
}

I_IV_V_minor = {
    "A Minor": ("D Minor", "E Minor"),
    "E Minor": ("A Minor", "B Minor"),
    "B Minor": ("E Minor", "F# Minor"),
    "F# Minor": ("B Minor", "C# Minor"),
    "C# Minor": ("F# Minor", "G# Minor"),
    "G# Minor": ("C# Minor", "D# Minor"),
    "D# Minor": ("G# Minor", "Bb Minor"),
    "Bb Minor": ("D# Minor", "F Minor"),
    "F Minor": ("Bb Minor", "C Minor"),
    "C Minor": ("F Minor", "G Minor"),
    "G Minor": ("C Minor", "D Minor"),
    "D Minor": ("G Minor", "A Minor")
}

def create_graph(include_relative=True):
    G = nx.Graph()
    
    # Add Circle of Fifths (Major)
    for i in range(len(circle_of_fifths_major)):
        G.add_edge(circle_of_fifths_major[i], circle_of_fifths_major[(i + 1) % len(circle_of_fifths_major)], 
                  label="Circle", relationship="Perfect fifth")

    # Add Circle of Fifths (Minor)
    for i in range(len(circle_of_fifths_minor)):
        G.add_edge(circle_of_fifths_minor[i], circle_of_fifths_minor[(i + 1) % len(circle_of_fifths_minor)], 
                  label="Circle", relationship="Perfect fifth")

    # Add I-IV-V relationships (Major)
    for chord, (IV, V) in I_IV_V_major.items():
        G.add_edge(chord, IV, label="I-IV", relationship="Perfect fourth")
        G.add_edge(chord, V, label="I-V", relationship="Perfect fifth")

    # Add I-IV-V relationships (Minor)
    for chord, (IV, V) in I_IV_V_minor.items():
        G.add_edge(chord, IV, label="I-IV", relationship="Perfect fourth")
        G.add_edge(chord, V, label="I-V", relationship="Perfect fifth")

    # Add suspended and diminished chords
    for chord, (sus2, sus4) in suspended_chords.items():
        G.add_edge(chord, sus2, label="Suspended", relationship="Major to sus2")
        G.add_edge(chord, sus4, label="Suspended", relationship="Major to sus4")

    for chord, dim in diminished_chords.items():
        G.add_edge(chord, dim, label="Diminished", relationship="Major to diminished")

    # Add relative major/minor relationships if enabled
    if include_relative:
        for major, (minor, transition) in relative_major_minor.items():
            G.add_edge(major, minor, label="Relative", relationship=transition)

    return G

def create_visualization(include_relative=True):
    G = create_graph(include_relative)
    pos = nx.spring_layout(G, dim=3)

    # Extract node positions
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    z_nodes = [pos[node][2] for node in G.nodes()]
    
    # Create node images
    node_images = []
    for node in G.nodes():
        node_images.append(create_chord_diagram_svg(node))

    # Extract edge positions and relationships
    x_edges = []
    y_edges = []
    z_edges = []
    edge_hover_texts = []
    
    for edge in G.edges(data=True):
        x_edges += [pos[edge[0]][0], pos[edge[1]][0], None]
        y_edges += [pos[edge[0]][1], pos[edge[1]][1], None]
        z_edges += [pos[edge[0]][2], pos[edge[1]][2], None]
        hover_text = f"{edge[2]['label']}: {edge[2]['relationship']}"
        edge_hover_texts.extend([hover_text, hover_text, None])

    # Create traces
    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='text',
        hovertext=edge_hover_texts,
        name='Connections'
    )

    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(
            symbol='circle',
            size=20,
            color='white',
            line=dict(color='blue', width=2)
        ),
        text=list(G.nodes()),
        textposition="bottom center",
        hovertemplate="%{text}<br><img src='%{customdata}' width=100><extra></extra>",
        customdata=node_images,
        name='Chords'
    )

    return [edge_trace, node_trace]

# Create the figure with both variants
traces_with_relative = create_visualization(True)
traces_without_relative = create_visualization(False)

# Create the figure with updatemenus (toggle button)
fig = go.Figure()

# Add both sets of traces
for trace in traces_with_relative:
    fig.add_trace(trace)

# Configure the updatemenus (toggle button)
updatemenus = [
    dict(
        type="buttons",
        direction="right",
        x=0.7,
        y=1.2,
        showactive=True,
        buttons=[
            dict(
                label="Show Relative Major/Minor",
                method="update",
                args=[{"visible": [True, True]},
                      {"title": "Chord Relationships (with Relative Major/Minor)"}]
            ),
            dict(
                label="Hide Relative Major/Minor",
                method="update",
                args=[{"visible": [True, True]},
                      {"title": "Chord Relationships (without Relative Major/Minor)"}],
                args2=[{"visible": traces_without_relative}]
            )
        ]
    )
]

# Update layout
fig.update_layout(
    title="3D Visualization of Harmonics",
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False),
    ),
    updatemenus=updatemenus,
    margin=dict(l=0, r=0, b=0, t=60),
    showlegend=False
)

fig.show()