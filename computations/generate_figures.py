import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
    from matplotlib.lines import Line2D
    import seaborn as sns
    from pathlib import Path

    # Set consistent style
    sns.set_theme(style="whitegrid")

    # Define output directory (MkDocs site assets for the online companion)
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"

    for _subdir in ["module1", "module6", "module7", "module8", "deep_dive"]:
        (OUTPUT_DIR / _subdir).mkdir(parents=True, exist_ok=True)

    # Create a modern color palette
    COLORS = {
        "black": 'k',
        "primary": "#2563eb",  # Blue
        "secondary": "#7c3aed",  # Purple
        "accent": "#f59e0b",  # Orange
        "success": "#10b981",  # Green
        "danger": "#ef4444",  # Red
        "neutral": "#6b7280",  # Gray
        "light": "#f3f4f6",  # Light gray
        "dark": "#1f2937",  # Dark gray
    }

    # Figure settings for slides
    FIG_WIDTH = 10
    FIG_HEIGHT = 6
    DPI = 150
    return (
        COLORS,
        Circle,
        DPI,
        FIG_HEIGHT,
        FIG_WIDTH,
        OUTPUT_DIR,
        mo,
        mpatches,
        np,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 1.1 Visualizations

    ## 1. ML Hierarchy (Nested Boxes)
    **Target slide**: `Module 1.1 - Introduction & Historical Context.md`, lines 25-38

    Shows the nested relationship: Computer Science > AI > ML > Deep Learning
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, mpatches, plt):
    # ML Hierarchy - Nested Boxes
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    _ax.set_xlim(0, 9)
    _ax.set_ylim(0, 6)
    _ax.axis("off")

    # Define boxes from outer to inner with distinct label positions
    _boxes = [
        {"xy": (0.5, 0.5), "width": 8, "height": 5, "label": "Computer Science", "color": COLORS["neutral"], "label_y": 5.3},
        {"xy": (1, 0.8), "width": 7, "height": 4, "label": "Artificial Intelligence", "color": COLORS["primary"], "label_y": 4.6},
        {"xy": (1.5, 1.1), "width": 6, "height": 3, "label": "Machine Learning", "color": COLORS["secondary"], "label_y": 3.9},
        {"xy": (2.0, 1.4), "width": 5, "height": 2, "label": "Deep Learning", "color": COLORS["black"], "label_y": 3.0},
    ]

    for _box in _boxes:
        _rect = mpatches.FancyBboxPatch(
            _box["xy"],
            _box["width"],
            _box["height"],
            boxstyle="round,pad=0.05,rounding_size=0.3",
            facecolor=_box["color"],
            edgecolor=_box["color"],
            linewidth=3,
            alpha=0.15,  # Reduced from 0.3 for clearer colors
        )
        _ax.add_patch(_rect)
        # Add label at distinct vertical position for each box
        _ax.text(
            _box["xy"][0] + _box["width"] / 2,
            _box["label_y"],
            _box["label"],
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=_box["color"],
        )

    # _ax.set_title("The AI/ML/Data Science Landscape", fontsize=16, fontweight="bold", pad=10)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "module1" / "ml_hierarchy.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'module1' / 'ml_hierarchy.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ML Task Categories Tree
    **Target slide**: `Module 1.1 - Introduction & Historical Context.md`, lines 99-110

    Shows the hierarchy: ML → Supervised/Unsupervised/Reinforcement → subtypes
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, plt):
    # ML Task Categories Tree
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    _ax.set_xlim(0, 10)
    _ax.set_ylim(1, 7)
    _ax.axis("off")

    # Node positions
    _nodes = {
        "ML": (5, 6),
        "Supervised\n(Labeled Data)": (2, 4),
        "Unsupervised\n(No Labels)": (5, 4),
        "Reinforcement\n(Feedback)": (8, 4),
        "Regression": (1.25, 2),
        "Classification": (2.75, 2),
        "Clustering": (4.25, 2),
        "Dimensionality Reduction": (5.75, 2),
    }

    # Draw edges
    _edges = [
        ("ML", "Supervised\n(Labeled Data)"),
        ("ML", "Unsupervised\n(No Labels)"),
        ("ML", "Reinforcement\n(Feedback)"),
        ("Supervised\n(Labeled Data)", "Regression"),
        ("Supervised\n(Labeled Data)", "Classification"),
        ("Unsupervised\n(No Labels)", "Clustering"),
        ("Unsupervised\n(No Labels)", "Dimensionality Reduction"),
    ]

    for _e1, _e2 in _edges:
        _x1, _y1 = _nodes[_e1]
        _x2, _y2 = _nodes[_e2]
        _ax.plot(
            [_x1, _x2],
            [_y1, _y2],
            color=COLORS["neutral"],
            linewidth=2,
            zorder=1,
        )

    # Draw nodes
    _node_colors = {
        "ML": COLORS["dark"],
        "Supervised\n(Labeled Data)": COLORS["primary"],
        "Unsupervised\n(No Labels)": COLORS["secondary"],
        "Reinforcement\n(Feedback)": COLORS["accent"],
        "Regression": COLORS["primary"],
        "Classification": COLORS["primary"],
        "Clustering": COLORS["secondary"],
        "Dimensionality Reduction": COLORS["secondary"],
    }

    for _name, (_x, _y) in _nodes.items():
        _color = _node_colors.get(_name, COLORS["neutral"])
        # Use larger radius for longer labels
        _radius = 0.6
        _circle = plt.Circle(
            (_x, _y),
            _radius,
            zorder=2,
            edgecolor='k', 
            facecolor=_color, 
            linewidth=2, 
            linestyle='-',
        )
        _ax.add_patch(_circle)
        # Split long labels for better fit
        if _name == 'Dimensionality Reduction':
            _display_name = 'Dimensionality\nReduction'
        else:
            _display_name = _name
        _ax.text(
            _x,
            _y,
            _display_name,
            ha="center",
            va="center",
            fontsize=8 if len(_name) > 10 else 9,
            fontweight="bold",
            color="white",
            zorder=3,
        )
        # # Add subtitle for main categories
        # if _name in _labels_below:
        #     _ax.text(
        #         _x,
        #         _y - 0.7,
        #         _labels_below[_name],
        #         ha="center",
        #         va="top",
        #         fontsize=9,
        #         color=COLORS["neutral"],
        #         style="italic",
        #     )

    # _ax.set_title("Machine Learning Task Categories", fontsize=16, fontweight="bold", pad=20)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "module1" / "ml_task_categories.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)

    print(f"Saved: {OUTPUT_DIR / 'module1' / 'ml_task_categories.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 1.3 Visualizations

    ## 3. Confusion Matrix
    **Target slide**: `Module 1.3 - Model Evaluation & Validation.md`, lines 78-88

    Shows the 2x2 confusion matrix with TP, TN, FP, FN
    """)
    return


@app.cell
def _(COLORS, DPI, OUTPUT_DIR, np, plt, sns):
    # Confusion Matrix
    _fig, _ax = plt.subplots(figsize=(5, 5))

    # Sample confusion matrix values
    _cm = np.array([[85, 15], [10, 90]])
    _labels = ["Positive", "Negative"]

    # Create heatmap
    sns.heatmap(
        _cm,
        annot=False,
        cmap="Blues",
        xticklabels=_labels,
        yticklabels=_labels,
        ax=_ax,
        cbar=False,
        linewidths=2,
        linecolor="k",
    )

    # Add custom annotations with labels
    # Use normalized values for color thresholding
    _max_val = _cm.max()
    _annotations = [
        (0, 0, "TP\n(True Positive)", _cm[0, 0]),
        (0, 1, "FN\n(False Negative)", _cm[0, 1]),
        (1, 0, "FP\n(False Positive)", _cm[1, 0]),
        (1, 1, "TN\n(True Negative)", _cm[1, 1]),
    ]

    for _row, _col, _label, _val in _annotations:
        # Use white text for cells with values above 40% of max (darker backgrounds)
        _text_color = "white" if _val > 0.4 * _max_val else COLORS["dark"]
        _ax.text(
            _col + 0.5,
            _row + 0.35,
            _label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=_text_color,
        )
        _ax.text(
            _col + 0.5,
            _row + 0.7,
            f"n = {_val}",
            ha="center",
            va="center",
            fontsize=11,
            color=_text_color,
        )

    _ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    _ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    # _ax.set_title("The Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    _ax.tick_params(axis="both", labelsize=11)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "module1" / "confusion_matrix.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'module1' / 'confusion_matrix.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. K-Fold Cross-Validation
    **Target slide**: `Module 1.3 - Model Evaluation & Validation.md`, lines 139-150

    Shows how data is split across 5 folds, with each fold taking turns as test set
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, plt):
    # K-Fold Cross-Validation
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT - 1))

    _n_folds = 5
    _n_splits = 5

    for _fold in range(_n_folds):
        for _split in range(_n_splits):
            _color = COLORS["accent"] if _split == _fold else COLORS["primary"]
            _label = "Test" if _split == _fold else "Train"
            _rect = plt.Rectangle(
                (_split * 1.8 + 0.5, _n_folds - _fold - 1),
                1.6,
                0.8,
                facecolor=_color,
                edgecolor="k",
                linewidth=2,
            )
            _ax.add_patch(_rect)
            if _label == "Test":
                _label_color='black'
            else:
                _label_color='white'
            _ax.text(
                _split * 1.8 + 1.3,
                _n_folds - _fold - 0.6,
                _label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=_label_color,
            )
        # Add fold label
        _ax.text(
            0.2,
            _n_folds - _fold - 0.6,
            f"Fold {_fold + 1}:",
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=COLORS["dark"],
        )

    # Add legend
    _legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["primary"], label="Training Data"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["accent"], label="Test Data"),
    ]

    _ax.set_xlim(-0.5, 10)
    _ax.set_ylim(-0.5, _n_folds + 0.5)
    _ax.axis("off")
    _ax.set_title("5-Fold Cross-Validation", fontsize=16, fontweight="bold", pad=20)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "module1" / "kfold_cv.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'module1' / 'kfold_cv.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Overfitting vs Underfitting
    **Target slide**: `Module 1.3 - Model Evaluation & Validation.md`, lines 153-161

    Shows three plots comparing underfitting, good fit, and overfitting
    """)
    return


@app.cell
def _(COLORS, DPI, OUTPUT_DIR, np, plt):
    # Overfitting vs Underfitting
    np.random.seed(42)

    # Generate data
    _x = np.linspace(0, 10, 50)
    _y_true = np.sin(_x) * 2 + 5
    _y_noisy = _y_true + np.random.normal(0, 0.5, len(_x))

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(8, 3),
    )

    _titles = ["Underfitting", "Good Fit", "Overfitting"]
    _subtitles = ["(High Bias)", "(Balanced)", "(High Variance)"]
    _colors_list = [COLORS["danger"], COLORS["success"], COLORS["accent"]]

    for _i, (_ax, _title, _subtitle, _color) in enumerate(
        zip(_axes, _titles, _subtitles, _colors_list)
    ):
        # Plot data points
        _ax.scatter(
            _x,
            _y_noisy,
            color=COLORS["primary"],
            alpha=0.6,
            s=30,
            zorder=2,
        )

        # Plot fitted line
        if _i == 0:  # Underfitting - straight line
            _y_fit = np.polyval(np.polyfit(_x, _y_noisy, 1), _x)
        elif _i == 1:  # Good fit - smooth curve
            _y_fit = np.polyval(np.polyfit(_x, _y_noisy, 5), _x)
        else:  # Overfitting - passes through every point
            from scipy.interpolate import make_interp_spline
            _x_smooth = np.linspace(_x.min(), _x.max(), 200)
            _spl = make_interp_spline(_x, _y_noisy, k=3)
            _y_fit = _spl(_x_smooth)
            _x = _x_smooth

        _ax.plot(
            _x if _i == 2 else np.linspace(0, 10, 50),
            _y_fit,
            color=_color,
            linewidth=3,
            zorder=3,
        )

        # Reset x for next iteration
        _x = np.linspace(0, 10, 50)

        _ax.set_title(f"{_title}\n{_subtitle}", fontsize=12, fontweight="bold")
        _ax.set_xlabel("")
        _ax.set_ylabel("")
        _ax.set_xticks([])
        _ax.set_yticks([])

    # _fig.suptitle("Overfitting vs Underfitting", fontsize=16, fontweight="bold", y=1.02)
    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "module1" / "overfitting_underfitting.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'module1' / 'overfitting_underfitting.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 6.1 Visualizations

    ## 6. Multi-Layer Perceptron (MLP) Architecture
    **Target slide**: `Module 6.1 - Introduction to Neural Networks.md`, lines 35-42

    Shows input layer → hidden layer(s) → output layer with connections
    """)
    return


@app.cell
def _(COLORS, Circle, DPI, OUTPUT_DIR, plt):
    # MLP Architecture
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.set_xlim(0, 10)
    _ax.set_ylim(1, 7)
    _ax.axis("off")

    # Layer positions
    _layers = {
        "input": {"x": 1.5, "neurons": 3, "color": COLORS["primary"], "label": "Input Layer"},
        "hidden1": {"x": 4, "neurons": 4, "color": COLORS["secondary"], "label": ""},  # No label, combined below
        "hidden2": {"x": 6.5, "neurons": 4, "color": COLORS["secondary"], "label": ""},
        "output": {"x": 9, "neurons": 1, "color": 'darkgrey', "label": "Output Layer"},
    }

    # Calculate y positions for neurons
    def _get_y_positions(_n_neurons, _center=3.5, _spacing=1.2):
        _start = _center - (_n_neurons - 1) * _spacing / 2
        return [_start + _i * _spacing for _i in range(_n_neurons)]

    _layer_positions = {}
    for _name, _info in _layers.items():
        _y_pos = _get_y_positions(_info["neurons"])
        _layer_positions[_name] = [(_info["x"], _y) for _y in _y_pos]

    # Draw connections
    _connection_pairs = [
        ("input", "hidden1"),
        ("hidden1", "hidden2"),
        ("hidden2", "output"),
    ]

    for _from_layer, _to_layer in _connection_pairs:
        for _x1, _y1 in _layer_positions[_from_layer]:
            for _x2, _y2 in _layer_positions[_to_layer]:
                _ax.plot(
                    [_x1, _x2],
                    [_y1, _y2],
                    color=COLORS["neutral"],
                    linewidth=0.8,
                    alpha=0.4,
                    zorder=1,
                )

    # Draw neurons
    for _name, _info in _layers.items():
        for _i, (_x, _y) in enumerate(_layer_positions[_name]):
            _circle = Circle(
                (_x, _y),
                0.35,
                edgecolor='k', 
                facecolor=_info["color"], 
                linewidth=2, 
                linestyle='-',
                zorder=2,
            )
            _ax.add_patch(_circle)

            # Add neuron labels for input layer (reverse order so x1 is at top)
            if _name == "input":
                _n_neurons = _layers[_name]["neurons"]
                _label_idx = _n_neurons - _i  # Reverse: top neuron gets x1
                _ax.text(
                    _x,
                    _y,
                    f"x{_label_idx}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    zorder=3,
                )
            elif _name == "output":
                _ax.text(
                    _x,
                    _y,
                    "ŷ",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    zorder=3,
                )

        # Add layer labels
        if _info["label"]:
            _ax.text(
                _info["x"],
                6.5,
                _info["label"],
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=_info["color"],
            )

    # Add hidden layers label spanning both
    _ax.text(
        5.25,
        6.5,
        "Hidden Layers",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=COLORS["secondary"],
    )

    # _ax.set_title("Multi-Layer Perceptron (MLP) Architecture", fontsize=16, fontweight="bold", pad=20)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "module6" / "mlp_architecture.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'module6' / 'mlp_architecture.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 7.2 Visualizations

    ## 7. CNN Pipeline (Conv → ReLU → Pool)
    **Target slide**: `Module 7.2 - Convolutional Neural Networks.md`, lines 77-79

    Shows the sequential pipeline of CNN operations
    """)
    return


@app.cell
def _(COLORS, DPI, OUTPUT_DIR, mpatches, plt):
    # CNN Pipeline
    _fig, _ax = plt.subplots(figsize=(10, 1.5))
    _ax.set_xlim(0, 12)
    _ax.set_ylim(0.75, 2.25)
    _ax.axis("off")

    # Pipeline stages
    _stages = [
        ("Conv", COLORS["primary"]),
        ("ReLU", COLORS["success"]),
        ("Pool", COLORS["secondary"]),
        ("Conv", COLORS["primary"]),
        ("ReLU", COLORS["success"]),
        ("Pool", COLORS["secondary"]),
        ("Flatten", COLORS["accent"]),
        ("FC", COLORS["danger"]),
    ]

    _x_start = 0.25
    _box_width = 1
    _spacing = 0.5

    for _i, (_label, _color) in enumerate(_stages):
        _x = _x_start + _i * (_box_width + _spacing)
        _rect = mpatches.FancyBboxPatch(
            (_x, 1),
            _box_width,
            1,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=_color,
            edgecolor="k",
            linewidth=2,
        )
        _ax.add_patch(_rect)
        _ax.text(
            _x + _box_width / 2,
            1.5,
            _label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )

        # Add arrow between stages (larger, more visible)
        if _i < len(_stages) - 1:
            _ax.annotate(
                "",
                xy=(_x + _box_width + _spacing - 0.05, 1.5),
                xytext=(_x + _box_width + 0.05, 1.5),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=COLORS["dark"],
                    lw=2,
                    mutation_scale=15,
                ),
            )

    # _ax.set_title("CNN Processing Pipeline", fontsize=16, fontweight="bold")

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "module7" / "cnn_pipeline.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'module7' / 'cnn_pipeline.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 8.2 Visualizations

    ## 8. RNN Unrolled Sequence
    **Target slide**: `Module 8.2 - Recurrent Neural Networks.md`, lines 29-31

    Shows RNN processing a sequence through time steps
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_WIDTH, OUTPUT_DIR, mpatches, plt):
    # RNN Unrolled Sequence
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, 4.5))
    _ax.set_xlim(0, 9)
    _ax.set_ylim(0.5, 4.5)
    _ax.axis("off")

    _n_steps = 4
    _x_start = 1
    _spacing = 2

    for _t in range(_n_steps):
        _x = _x_start + _t * _spacing

        # Input node
        _input_circle = plt.Circle(
            (_x, 1),
            0.4,
            facecolor=COLORS["primary"],
            edgecolor="k",
            linewidth=2,
        )
        _ax.add_patch(_input_circle)
        _ax.text(
            _x,
            1.0,
            f"x{_t + 1}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

        # RNN cell
        _rect = mpatches.FancyBboxPatch(
            (_x - 0.5, 2),
            1,
            1,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=COLORS["secondary"],
            edgecolor="k",
            linewidth=2,
        )
        _ax.add_patch(_rect)
        _ax.text(
            _x,
            2.5,
            f"h{_t + 1}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

        # Arrow from input to RNN
        _ax.annotate(
            "",
            xy=(_x, 2),
            xytext=(_x, 1.4),
            arrowprops=dict(
                arrowstyle="->",
                color=COLORS["neutral"],
                lw=2,
            ),
        )

        # Hidden state node
        _hidden_circle = plt.Circle(
            (_x, 4),
            0.35,
            facecolor=COLORS["accent"],
            edgecolor="k",
            linewidth=2,
        )
        _ax.add_patch(_hidden_circle)
        _ax.text(
            _x,
            4,
            f"y{_t + 1}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

        # Arrow from RNN to hidden
        _ax.annotate(
            "",
            xy=(_x, 3.7),
            xytext=(_x, 3),
            arrowprops=dict(
                arrowstyle="->",
                color=COLORS["neutral"],
                lw=2,
            ),
        )

        # Arrow to next RNN (hidden state connection)
        if _t < _n_steps - 1:
            _ax.annotate(
                "",
                xy=(_x + _spacing - 0.5, 2.5),
                xytext=(_x + 0.5, 2.5),
                arrowprops=dict(
                    arrowstyle="->",
                    color=COLORS["accent"],
                    lw=2,
                    connectionstyle="arc3,rad=0",
                ),
            )

    # Output arrow from last hidden state (more prominent)
    _ax.annotate(
        "",
        xy=(8.5, 2.5),
        xytext=(7.5, 2.5),
        arrowprops=dict(
            arrowstyle="-|>",
            color=COLORS["danger"],
            lw=3,
            mutation_scale=15,
        ),
    )
    # _ax.text(
    #     11.5,
    #     4.3,
    #     "Output",
    #     ha="left",
    #     va="center",
    #     fontsize=13,
    #     fontweight="bold",
    #     color=COLORS["danger"],
    # )

    # _ax.set_title("RNN Unrolled Through Time", fontsize=16, fontweight="bold", pad=20)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "module8" / "rnn_unrolled.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'module8' / 'rnn_unrolled.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Deep Dive - Universal Approximators Visualizations

    ## 9. Sigmoid Function Curves
    **Target slide**: `Deep Dive - Universal Approximators.md`, lines 81-89

    Shows how sigmoid creates S-shaped curves with different weights
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_WIDTH, OUTPUT_DIR, np, plt):
    # Sigmoid Function Curves
    def _sigmoid(_x, _weight=1, _bias=0):
        return 1 / (1 + np.exp(-_weight * (_x - _bias)))

    _x = np.linspace(-6, 6, 200)

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(FIG_WIDTH, 4),
    )

    # Standard sigmoid
    _axes[0].plot(
        _x,
        _sigmoid(_x, _weight=1),
        color=COLORS["primary"],
        linewidth=3,
    )
    _axes[0].axhline(y=0.5, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    _axes[0].axvline(x=0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    _axes[0].set_title("Standard Sigmoid\n(weight = 1)", fontsize=12, fontweight="bold")
    _axes[0].set_xlabel("x")
    _axes[0].set_ylabel("σ(x)")
    _axes[0].set_ylim(-0.1, 1.1)

    # Sharper sigmoid (larger weight)
    _axes[1].plot(
        _x,
        _sigmoid(_x, _weight=5),
        color=COLORS["accent"],
        linewidth=3,
    )
    _axes[1].axhline(y=0.5, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    _axes[1].axvline(x=0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    _axes[1].set_title("Sharper Sigmoid\n(weight = 5)", fontsize=12, fontweight="bold")
    _axes[1].set_xlabel("x")
    _axes[1].set_ylabel("σ(x)")
    _axes[1].set_ylim(-0.1, 1.1)

    _fig.suptitle("Sigmoid Activation Functions", fontsize=16, fontweight="bold", y=1.02)
    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "sigmoid_curves.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'sigmoid_curves.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Building Complex Functions with Multiple Neurons
    **Target slide**: `Deep Dive - Universal Approximators.md`, lines 101-108

    Shows how combining neurons approximates complex shapes
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_WIDTH, OUTPUT_DIR, np, plt):
    # Building Complex Functions
    def _sigmoid_fn(_x, _w=1, _b=0):
        return 1 / (1 + np.exp(-_w * (_x - _b)))

    def _bump(_x, _center, _width, _height=1):
        """Create a bump function using two sigmoids"""
        _left = _sigmoid_fn(_x, _w=10, _b=_center - _width / 2)
        _right = _sigmoid_fn(_x, _w=10, _b=_center + _width / 2)
        return _height * (_left - _right)

    _x = np.linspace(-1, 11, 500)

    # Target function (two bumps)
    _target = _bump(_x, 3, 2, 1) + _bump(_x, 8, 2, 1)

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(FIG_WIDTH, 3),
    )

    # Target shape
    _axes[0].fill_between(
        _x,
        _target,
        alpha=0.3,
        color=COLORS["primary"],
    )
    _axes[0].plot(
        _x,
        _target,
        color=COLORS["primary"],
        linewidth=2,
    )
    _axes[0].set_title("Target Shape", fontsize=12, fontweight="bold")
    _axes[0].set_ylim(-0.2, 1.3)

    # 2 neurons (rough approximation) - made much wider and less precise
    _approx_2 = _bump(_x, 5.5, 7.5, 0.7)  # Single wide bump, clearly different from target
    _axes[1].fill_between(
        _x,
        _target,
        alpha=0.2,
        color=COLORS["primary"],
        label="Target",
    )
    _axes[1].plot(
        _x,
        _approx_2,
        color=COLORS["accent"],
        linewidth=3,
        label="Approximation",
    )
    _axes[1].set_title("2 Neurons\n(Rough Approximation)", fontsize=12, fontweight="bold")
    _axes[1].set_ylim(-0.2, 1.3)
    _axes[1].legend(fontsize=8, loc="upper right")

    # 10 neurons (better approximation)
    _axes[2].fill_between(
        _x,
        _target,
        alpha=0.2,
        color=COLORS["primary"],
    )
    _axes[2].plot(
        _x,
        _target,
        color=COLORS["success"],
        linewidth=2,
    )
    _axes[2].set_title("10 Neurons\n(Better Approximation)", fontsize=12, fontweight="bold")
    _axes[2].set_ylim(-0.2, 1.3)

    for _ax in _axes:
        _ax.set_xlabel("")
        _ax.set_xticks([])
        _ax.set_yticks([])

    # _fig.suptitle("Universal Approximation: More Neurons = Better Fit", fontsize=16, fontweight="bold", y=1.02)
    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "neuron_approximation.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'neuron_approximation.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. ReLU Function
    **Target slide**: `Deep Dive - Universal Approximators.md`, lines 367-373

    Shows the ReLU activation function shape
    """)
    return


@app.cell
def _(COLORS, DPI, OUTPUT_DIR, np, plt):
    # ReLU Function
    _x = np.linspace(-3, 3, 200)
    _relu = np.maximum(0, _x)

    _fig, _ax = plt.subplots(figsize=(6, 4))

    _ax.plot(
        _x,
        _relu,
        color=COLORS["primary"],
        linewidth=3,
    )
    _ax.axhline(y=0, color=COLORS["neutral"], linestyle="-", alpha=0.3)
    _ax.axvline(x=0, color=COLORS["neutral"], linestyle="-", alpha=0.3)

    # Highlight the "kink"
    # _ax.scatter(
    #     [0],
    #     [0],
    #     color=COLORS["accent"],
    #     s=100,
    #     zorder=5,
    # )
    # _ax.annotate(
    #     "Kink at x=0",
    #     xy=(0, 0),
    #     xytext=(1, 1),
    #     fontsize=11,
    #     arrowprops=dict(
    #         arrowstyle="->",
    #         color=COLORS["accent"],
    #     ),
    # )

    _ax.set_xlabel("x", fontsize=12)
    _ax.set_ylabel("ReLU(x) = max(0, x)", fontsize=12)
    # _ax.set_title("ReLU Activation Function", fontsize=16, fontweight="bold", pad=20)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "relu_function.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'relu_function.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. ReLU Bump Function (Two ReLUs Combined)
    **Target slide**: `Deep Dive - Universal Approximators.md`, lines 382-391

    Shows how two ReLUs can create a "bump" shape
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_WIDTH, OUTPUT_DIR, np, plt):
    # ReLU Bump Function - extended x range to show the plateau clearly
    _x = np.linspace(-1, 4, 250)
    _relu1 = np.maximum(0, _x)
    _relu2 = np.maximum(0, _x - 1)
    _bump = _relu1 - _relu2

    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(FIG_WIDTH, 3.5),
    )

    # ReLU(x)
    _axes[0].plot(
        _x,
        _relu1,
        color=COLORS["primary"],
        linewidth=3,
    )
    _axes[0].axhline(y=0, color=COLORS["neutral"], linestyle="-", alpha=0.3)
    _axes[0].axvline(x=0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    _axes[0].set_title("ReLU(x)", fontsize=12, fontweight="bold")
    _axes[0].set_ylim(-0.5, 4)

    # ReLU(x-1)
    _axes[1].plot(
        _x,
        _relu2,
        color=COLORS["secondary"],
        linewidth=3,
    )
    _axes[1].axhline(y=0, color=COLORS["neutral"], linestyle="-", alpha=0.3)
    _axes[1].axvline(x=0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    _axes[1].set_title("ReLU(x - 1)", fontsize=12, fontweight="bold")
    _axes[1].set_ylim(-0.5, 4)

    # Combined - THIS shows the bump shape: 0 -> rise -> plateau at 1
    _axes[2].fill_between(
        _x,
        _bump,
        alpha=0.3,
        color=COLORS["accent"],
    )
    _axes[2].plot(
        _x,
        _bump,
        color=COLORS["accent"],
        linewidth=3,
    )
    _axes[2].axhline(y=0, color=COLORS["neutral"], linestyle="-", alpha=0.3)
    _axes[2].axhline(y=1, color=COLORS["accent"], linestyle="--", alpha=0.5, linewidth=1)
    _axes[2].axvline(x=0, color=COLORS["neutral"], linestyle="--", alpha=0.3)
    _axes[2].set_title("ReLU(x) - ReLU(x-1)", fontsize=12, fontweight="bold")
    _axes[2].set_ylim(-0.5, 4)  # Focus on the bump shape

    # _fig.suptitle("Creating a Ramp with Two ReLUs", fontsize=16, fontweight="bold", y=1.02)
    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "relu_bump.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'relu_bump.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. XOR Decision Boundaries
    **Target slide**: `Deep Dive - Universal Approximators.md`, lines 649-660

    Shows linear vs non-linear decision boundaries for XOR problem
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, np, plt):
    # XOR Decision Boundaries
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(FIG_WIDTH, FIG_HEIGHT - 1),
    )

    # XOR data points
    _xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    _xor_y = np.array([0, 1, 1, 0])

    # Plot function
    def _plot_xor(_ax, _title, _linear=True):
        # Plot points
        for _i, (_point, _label) in enumerate(zip(_xor_x, _xor_y)):
            _color = COLORS["primary"] if _label == 0 else COLORS["accent"]
            _marker = "o" if _label == 0 else "s"
            _ax.scatter(
                _point[0],
                _point[1],
                color=_color,
                s=200,
                marker=_marker,
                edgecolor="white",
                linewidth=2,
                zorder=3,
            )

        # Draw decision boundary
        if _linear:
            # Linear boundary (fails) - diagonal line shows it can't separate
            _ax.plot([-0.3, 1.3], [-0.3, 1.3], color=COLORS["danger"], linewidth=2, linestyle="--")
            # Add X marks to show misclassification
            _ax.text(0.5, -0.35, "Cannot separate!", fontsize=10, ha="center", color=COLORS["danger"])
        else:
            # Non-linear boundary (succeeds) - two parallel diagonal lines
            # Neural networks solve XOR with piecewise linear boundaries
            # Line 1: x₁ + x₂ = 0.5 (separates (0,0) from class 1 points)
            # Line 2: x₁ + x₂ = 1.5 (separates (1,1) from class 1 points)
            _t = np.linspace(-0.3, 1.3, 100)
            _ax.plot(_t, 0.5 - _t, color=COLORS["success"], linewidth=2.5)
            _ax.plot(_t, 1.5 - _t, color=COLORS["success"], linewidth=2.5)
            # Shade the regions to show the classes
            # Class 0 (blue): below line 1 (lower-left) and above line 2 (upper-right)
            _ax.fill([-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0], alpha=0.1, color=COLORS["primary"])
            _ax.fill([0, 1.5, 1.5], [1.5, 1.5, 0], alpha=0.1, color=COLORS["primary"])
            # Class 1 (orange): the band between the two lines
            _ax.fill([-0.5, 1.0, 1.5, 1.5, 0, -0.5], [1.0, -0.5, -0.5, 0, 1.5, 1.5], alpha=0.1, color=COLORS["accent"])

        _ax.set_xlim(-0.5, 1.5)
        _ax.set_ylim(-0.5, 1.5)
        _ax.set_aspect("equal")
        _ax.set_title(_title, fontsize=12, fontweight="bold")
        _ax.set_xlabel(r"$x_1$")
        _ax.set_ylabel(r"$x_2$")

    _plot_xor(_axes[0], "Logistic Regression\n(Linear Boundary)\nAccuracy: 50%", _linear=True)
    _plot_xor(_axes[1], "Neural Network\n(Non-Linear Boundary)\nAccuracy: 100%", _linear=False)

    # Legend
    _legend_elements = [
        plt.scatter([], [], color=COLORS["primary"], s=100, marker="o", label="Class 0"),
        plt.scatter([], [], color=COLORS["accent"], s=100, marker="s", label="Class 1"),
    ]
    _axes[1].legend(
        handles=_legend_elements,
        loc="upper right",
        fontsize=10,
    )

    _fig.suptitle("XOR Problem: Why Neural Networks Win", fontsize=16, fontweight="bold", y=1.02)
    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "xor_decision_boundary.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'xor_decision_boundary.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Model Complexity Spectrum
    **Target slide**: `Deep Dive - Universal Approximators.md`, lines 551-561

    Shows the spectrum from simple to complex models
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, plt):
    # Model Complexity Spectrum
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT - 2))
    _ax.set_xlim(0, 12)
    _ax.set_ylim(0.35, .6)
    _ax.axis("off")

    # Draw spectrum arrow
    _ax.annotate(
        "",
        xy=(11, 0.5),
        xytext=(1, 0.5),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["dark"],
            lw=3,
        ),
    )

    # Labels at ends
    _ax.text(
        0.5,
        0.5,
        "Simple",
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["primary"],
    )
    _ax.text(
        11.5,
        0.5,
        "Complex",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["danger"],
    )

    # Models on spectrum
    _models = [
        (1.5, "Linear\nRegression", COLORS["primary"]),
        (3.5, "Logistic\nRegression", COLORS["primary"]),
        (5.5, "Polynomial\nRegression", COLORS["secondary"]),
        (7.5, "Decision\nTree", COLORS["secondary"]),
        (9.5, "Deep\nNeural Net", COLORS["danger"]),
    ]

    for _x, _label, _color in _models:
        # Marker
        _ax.scatter(
            [_x],
            [0.5],
            s=300,
            color=_color,
            zorder=3,
        )
        # Label
        _ax.text(
            _x,
            0.4,
            _label,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=_color,
        )

    # _ax.set_title("Model Complexity Spectrum", fontsize=16, fontweight="bold", pad=10)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "model_spectrum.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'model_spectrum.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Deep Dive - CNN Architecture Visualizations

    ## 15. Local Connectivity vs Fully Connected
    **Target slide**: `Deep Dive - CNN Architecture.md`, lines 191-201

    Shows the difference between fully connected and locally connected architectures
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, plt):
    # Local Connectivity Visualization
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(FIG_WIDTH, FIG_HEIGHT - 1),
    )

    def _draw_grid(_ax, _size, _highlight=None):
        """Draw a grid of neurons"""
        for _i in range(_size):
            for _j in range(_size):
                _is_highlight = (_highlight and
                                 _highlight[0] <= _i < _highlight[0] + _highlight[2] and
                                 _highlight[1] <= _j < _highlight[1] + _highlight[3])
                _color = COLORS["accent"] if _is_highlight else COLORS["light"]
                _edge = COLORS["accent"] if _is_highlight else COLORS["neutral"]
                _circle = plt.Circle(
                    (_j, _size - 1 - _i),
                    0.35,
                    facecolor=_color,
                    edgecolor=_edge,
                    linewidth=2,
                )
                _ax.add_patch(_circle)

    # Fully Connected
    _draw_grid(_axes[0], 5)
    _axes[0].set_xlim(-1, 8.5)  # Extended to show output neuron at x=7
    _axes[0].set_ylim(-1, 5.5)
    _axes[0].set_aspect("equal")
    _axes[0].axis("off")

    # Add output neuron with all connections
    _axes[0].scatter([7], [2], s=1000, color=COLORS["primary"], zorder=3)
    _axes[0].text(
        7,
        2,
        "Out",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )
    # Draw connections from all input neurons
    for _i in range(5):
        for _j in range(5):
            _axes[0].plot(
                [_j, 5.5],
                [4 - _i, 2],
                color=COLORS["neutral"],
                alpha=0.8,
                linewidth=2,
            )
    _axes[0].annotate(
        "",
        xy=(6.6, 2),
        xytext=(5.5, 2),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["neutral"],
            lw=2,
        ),
    )
    _axes[0].set_title("Fully Connected\n(25 connections)", fontsize=12, fontweight="bold")

    # Local Connectivity
    _draw_grid(_axes[1], 5, _highlight=(1, 1, 3, 3))
    _axes[1].set_xlim(-1, 8.5)  # Extended to show output neuron at x=7
    _axes[1].set_ylim(-1, 5.5)
    _axes[1].set_aspect("equal")
    _axes[1].axis("off")

    # Add output neuron with local connections
    _axes[1].scatter([7], [2], s=1000, color=COLORS["primary"], zorder=3)
    _axes[1].text(
        7,
        2,
        "Out",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )
    # Draw connections only from highlighted region
    for _i in range(1, 4):
        for _j in range(1, 4):
            _axes[1].plot(
                [_j, 5.5],
                [4 - _i, 2],
                color=COLORS["accent"],
                alpha=0.8,
                linewidth=2,
            )
    _axes[1].annotate(
        "",
        xy=(6.6, 2),
        xytext=(5.5, 2),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["accent"],
            lw=2,
        ),
    )
    _axes[1].set_title("Local Connectivity\n(9 connections)", fontsize=12, fontweight="bold")

    # _fig.suptitle("Fully Connected vs Local Connectivity", fontsize=16, fontweight="bold", y=1.02)
    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "local_connectivity.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'local_connectivity.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. Max Pooling Operation
    **Target slide**: `Deep Dive - CNN Architecture.md`, lines 524-533

    Shows 2x2 max pooling reducing a 4x4 input to 2x2 output
    """)
    return


@app.cell
def _(COLORS, DPI, OUTPUT_DIR, np, plt, sns):
    # Max Pooling Operation
    _fig, _axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [1, 0.5, 1]},
    )

    # Input matrix
    _input_data = np.array([
        [1, 3, 2, 4],
        [0, 2, 1, 3],
        [5, 6, 7, 8],
        [4, 1, 5, 2],
    ])

    # Output matrix (max pooled)
    _output_data = np.array([
        [3, 4],
        [6, 8],
    ])

    # Draw input with pooling regions highlighted
    sns.heatmap(
        _input_data,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=_axes[0],
        cbar=False,
        linewidths=2,
        linecolor="k",
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    # Add thick lines to show 2x2 pooling regions
    _axes[0].axhline(y=2, color=COLORS["accent"], linewidth=4)
    _axes[0].axvline(x=2, color=COLORS["accent"], linewidth=4)
    _axes[0].set_title("Input (4×4)", fontsize=12, fontweight="bold")
    _axes[0].set_xticks([])
    _axes[0].set_yticks([])

    # Arrow
    _axes[1].axis("off")
    _axes[1].text(
        0.5,
        0.5,
        "Max Pool\n2×2\n→",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["accent"],
    )

    # Draw output
    sns.heatmap(
        _output_data,
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=_axes[2],
        cbar=False,
        linewidths=2,
        linecolor="k",
        annot_kws={"fontsize": 16, "fontweight": "bold"},
    )
    _axes[2].set_title("Output (2×2)", fontsize=12, fontweight="bold")
    _axes[2].set_xticks([])
    _axes[2].set_yticks([])

    # _fig.suptitle("Max Pooling: Keep Maximum in Each Region", fontsize=16, fontweight="bold", y=1.02)
    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "max_pooling.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'max_pooling.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Deep Dive - Transformer Architecture Visualizations

    ## 17. Self-Attention Dimension Flow 1
    **Target slide**: `Deep Dive - Transformer Architecture.md`, lines 491-519

    Shows the tensor dimensions flowing through self-attention computation
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_WIDTH, OUTPUT_DIR, mpatches, plt):
    # Self-Attention Dimension Flow
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, 6))
    _ax.set_xlim(0, 10)
    _ax.set_ylim(7, 15.75)
    _ax.axis("off")

    # Define data tensor nodes (solid fill)
    # Flow: Embeddings -> +Pos Enc -> X -> W matrices -> Q,K,V -> Attention -> Output
    _data_nodes = [
        (5, 15, "Embeddings\n(seq × d_model)", COLORS["primary"]),
        (5, 13.5, "+ Positional Encoding\n(seq × d_model)", COLORS["primary"]),
        (5, 12, "X\n(seq × d_model)", COLORS["primary"]),
        (2, 8, "Q\n(seq × d_model)", COLORS["secondary"]),
        (5, 8, "K\n(seq × d_model)", COLORS["secondary"]),
        (8, 8, "V\n(seq × d_model)", COLORS["secondary"]),
    ]

    # Define learnable weight matrix nodes (hatched fill)
    _weight_nodes = [
        (2, 10, "x W_Q\n(d_model × d_model)", COLORS["danger"]),
        (5, 10, "x W_K\n(d_model × d_model)", COLORS["danger"]),
        (8, 10, "x W_V\n(d_model × d_model)", COLORS["danger"]),
    ]

    # Draw data tensor nodes (solid fill)
    for _x, _y, _label, _color in _data_nodes:
        _rect = mpatches.FancyBboxPatch(
            (_x - 1, _y - 0.5),
            2,
            1.0,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=_color,
            edgecolor="k",
            linewidth=2,
        )
        _ax.add_patch(_rect)
        _ax.text(
            _x,
            _y,
            _label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    # Draw learnable weight matrix nodes (hatched fill to indicate learnable)
    for _x, _y, _label, _color in _weight_nodes:
        # Background solid color
        _rect_bg = mpatches.FancyBboxPatch(
            (_x - 1, _y - 0.5),
            2,
            1.0,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=_color,
            edgecolor="k",
            linewidth=2,
        )
        _ax.add_patch(_rect_bg)
        # Hatched overlay to indicate learnable parameters
        _rect_hatch = mpatches.FancyBboxPatch(
            (_x - 1, _y - 0.5),
            2,
            1.0,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="none",
            edgecolor="k",
            linewidth=2,
            hatch="///",
            alpha=0.2,
        )
        _ax.add_patch(_rect_hatch)
        _ax.text(
            _x,
            _y,
            _label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    # Add "learnable" annotation
    _ax.text(
        9.5,
        10,
        "← Learnable!",
        ha="left",
        va="center",
        fontsize=12,
        color='k',
        style="italic",
    )

    # Draw arrows
    _arrows = [
        # Embeddings to Positional Encoding
        ((5, 14.5), (5, 14)),
        # Positional Encoding to X
        ((5, 13), (5, 12.5)),
        # X to weight matrices
        ((5, 11.5), (2, 10.5)),
        ((5, 11.5), (5, 10.5)),
        ((5, 11.5), (8, 10.5)),
        # Weight matrices to Q, K, V
        ((2, 9.5), (2, 8.5)),
        ((5, 9.5), (5, 8.5)),
        ((8, 9.5), (8, 8.5)),
    ]

    for (_x1, _y1), (_x2, _y2) in _arrows:
        _ax.annotate(
            "",
            xy=(_x2, _y2),
            xytext=(_x1, _y1),
            arrowprops=dict(
                arrowstyle="->",
                color=COLORS["neutral"],
                lw=1.5,
            ),
        )

    # _ax.set_title("Self-Attention: Dimension Flow", fontsize=16, fontweight="bold", pad=20)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "attention_flow-1.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'attention_flow-1.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 18. Self-Attention Dimension Flow 2
    **Target slide**: `Deep Dive - Transformer Architecture.md`, lines 491-519

    Shows the tensor dimensions flowing through self-attention computation
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_WIDTH, OUTPUT_DIR, mpatches, plt):
    # Self-Attention Dimension Flow
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, 6))
    _ax.set_xlim(0, 10)
    _ax.set_ylim(-1, 9)
    _ax.axis("off")

    # Define data tensor nodes (solid fill)
    # Flow: Embeddings -> +Pos Enc -> X -> W matrices -> Q,K,V -> Attention -> Output
    _data_nodes = [
        (2, 8, "Q\n(seq × d_model)", COLORS["secondary"]),
        (5, 8, "K\n(seq × d_model)", COLORS["secondary"]),
        (8, 8, "V\n(seq × d_model)", COLORS["secondary"]),
        (3.5, 5.5, r"Q @ $K^T$ / $\sqrt{d}$" + "\n(seq × seq)", COLORS["accent"]),
        (3.5, 3.5, "softmax\n(seq × seq)", COLORS["accent"]),
        (5, 1.5, "weights @ V\n(seq × d_model)", COLORS["accent"]),
        (5, 0, "Output\n(seq × d_model)", COLORS["success"]),
    ]

    # Define learnable weight matrix nodes (hatched fill)
    _weight_nodes = []

    # Draw data tensor nodes (solid fill)
    for _x, _y, _label, _color in _data_nodes:
        _rect = mpatches.FancyBboxPatch(
            (_x - 1, _y - 0.5),
            2,
            1.0,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=_color,
            edgecolor="k",
            linewidth=2,
        )
        _ax.add_patch(_rect)
        _ax.text(
            _x,
            _y,
            _label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    # Draw learnable weight matrix nodes (hatched fill to indicate learnable)
    for _x, _y, _label, _color in _weight_nodes:
        # Background solid color
        _rect_bg = mpatches.FancyBboxPatch(
            (_x - 1, _y - 0.5),
            2,
            1.0,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=_color,
            edgecolor="k",
            linewidth=2,
        )
        _ax.add_patch(_rect_bg)
        # Hatched overlay to indicate learnable parameters
        _rect_hatch = mpatches.FancyBboxPatch(
            (_x - 1, _y - 0.5),
            2,
            1.0,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="none",
            edgecolor="k",
            linewidth=2,
            hatch="///",
            alpha=0.2,
        )
        _ax.add_patch(_rect_hatch)
        _ax.text(
            _x,
            _y,
            _label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    # Draw arrows
    _arrows = [
        # Q, K to attention
        ((2, 7.5), (3.5, 6)),
        ((5, 7.5), (3.5, 6)),
        # Attention to softmax
        ((3.5, 5), (3.5, 4)),
        # Softmax and V to weighted sum
        ((3.5, 3), (5, 2)),
        ((8, 7.5), (5, 2)),
        # Weighted sum to output
        ((5, 1), (5, 0.5)),
    ]

    for (_x1, _y1), (_x2, _y2) in _arrows:
        _ax.annotate(
            "",
            xy=(_x2, _y2),
            xytext=(_x1, _y1),
            arrowprops=dict(
                arrowstyle="->",
                color=COLORS["neutral"],
                lw=1.5,
            ),
        )

    #_ax.set_title("Self-Attention: Dimension Flow", fontsize=16, fontweight="bold", pad=20)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "attention_flow-2.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'attention_flow-2.png'}")
    return


@app.cell
def _(OUTPUT_DIR):
    OUTPUT_DIR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Deep Dive: Surprising Phenomena in Modern Deep Learning

    ## 1. Bias-Variance U-Curve
    **Target slide**: `Deep Dive - Surprising Phenomena in Modern Deep Learning.md`, lines 45-59

    Shows the classical U-shaped curve of test error vs model complexity with underfitting/overfitting zones.
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, np, plt):
    # Bias-Variance U-Curve
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT - 1))

    # Generate U-shaped curve data
    _x = np.linspace(0, 10, 200)
    _y = 0.5 + 2.5 * ((_x - 5) / 5) ** 2  # U-curve with minimum at x=5

    # Plot the curve
    _ax.plot(_x, _y, color=COLORS["primary"], linewidth=3, zorder=3)

    # Shade underfitting region (left) - full height background
    _ax.axvspan(0, 3, alpha=0.15, color=COLORS["primary"], zorder=1)

    # Shade overfitting region (right) - full height background
    _ax.axvspan(7, 10, alpha=0.15, color=COLORS["danger"], zorder=1)

    # Mark the sweet spot
    _sweet_spot_x = 5
    _sweet_spot_y = 0.5
    _ax.scatter(
        [_sweet_spot_x],
        [_sweet_spot_y],
        color=COLORS["success"],
        s=150,
        zorder=4,
        marker="*",
        edgecolors="white",
        linewidths=1,
    )
    _ax.axvline(
        x=_sweet_spot_x,
        color=COLORS["success"],
        linestyle="--",
        alpha=0.5,
        linewidth=1.5,
    )

    # Add annotations
    _ax.annotate(
        "High Bias\n(Underfitting)",
        xy=(1.5, 2.0),
        fontsize=12,
        ha="center",
        color=COLORS["primary"],
        fontweight="bold",
    )
    _ax.annotate(
        "High Variance\n(Overfitting)",
        xy=(8.5, 2.0),
        fontsize=12,
        ha="center",
        color=COLORS["danger"],
        fontweight="bold",
    )
    _ax.annotate(
        "Sweet Spot",
        xy=(_sweet_spot_x, _sweet_spot_y),
        xytext=(_sweet_spot_x + 1.5, _sweet_spot_y + 0.8),
        fontsize=11,
        ha="left",
        color=COLORS["success"],
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["success"],
            lw=1.5,
        ),
    )

    # Style the axes
    _ax.set_xlabel("Model Complexity", fontsize=12, fontweight="bold")
    _ax.set_ylabel("Test Error", fontsize=12, fontweight="bold")
    _ax.set_xlim(0, 10)
    _ax.set_ylim(0, 3.5)
    _ax.set_xticks([])
    _ax.set_yticks([])

    # Add arrow indicators on axes
    _ax.annotate(
        "",
        xy=(10, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=COLORS["dark"], lw=1.5),
    )
    _ax.annotate(
        "",
        xy=(0, 3.5),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=COLORS["dark"], lw=1.5),
    )

    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.spines["left"].set_visible(False)
    _ax.spines["bottom"].set_visible(False)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "bias_variance_u_curve.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'bias_variance_u_curve.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Early Stopping
    **Target slide**: `Deep Dive - Surprising Phenomena in Modern Deep Learning.md`, lines 112-130

    Shows training and validation loss curves over epochs with the optimal stopping point marked.
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, np, plt):
    # Early Stopping
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT - 1))

    # Generate epoch data
    _epochs = np.linspace(0, 100, 200)

    # Training loss: exponential decay to plateau
    _train_loss = 1.8 * np.exp(-0.05 * _epochs) + 0.15

    # Validation loss: drops then rises (U-shaped)
    _val_loss = 1.8 * np.exp(-0.06 * _epochs) + 0.25 + 0.008 * np.maximum(0, _epochs - 40)

    # Find optimal stopping point (minimum validation loss)
    _stop_idx = np.argmin(_val_loss)
    _stop_epoch = _epochs[_stop_idx]
    _stop_val = _val_loss[_stop_idx]

    # Plot curves
    _ax.plot(
        _epochs,
        _train_loss,
        color=COLORS["primary"],
        linewidth=2.5,
        label="Training Loss",
    )
    _ax.plot(
        _epochs,
        _val_loss,
        color=COLORS["accent"],
        linewidth=2.5,
        label="Validation Loss",
    )

    # Mark optimal stopping point
    _ax.axvline(
        x=_stop_epoch,
        color=COLORS["success"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    _ax.scatter(
        [_stop_epoch],
        [_stop_val],
        color=COLORS["success"],
        s=120,
        zorder=5,
        marker="o",
        edgecolors="white",
        linewidths=2,
    )

    # Shade overfitting region
    _ax.axvspan(
        _stop_epoch,
        100,
        alpha=0.1,
        color=COLORS["danger"],
    )

    # Add "Stop here!" annotation
    _ax.annotate(
        "Stop here!",
        xy=(_stop_epoch, _stop_val),
        xytext=(_stop_epoch + 15, _stop_val + 0.3),
        fontsize=12,
        ha="left",
        color=COLORS["success"],
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["success"],
            lw=2,
        ),
    )

    # Add overfitting label - positioned at top for better contrast
    _ax.annotate(
        "Overfitting\nzone",
        xy=(75, 1.7),
        fontsize=11,
        ha="center",
        color="#b91c1c",  # Darker red for better contrast on light red background
        fontweight="bold",
    )

    # Style
    _ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
    _ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    _ax.set_xlim(0, 100)
    _ax.set_ylim(0, 2)
    _ax.legend(loc="upper right", fontsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "early_stopping.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'early_stopping.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Double Descent
    **Target slide**: `Deep Dive - Surprising Phenomena in Modern Deep Learning.md`, lines 151-169

    Shows the double descent phenomenon: test error follows classical U-curve, peaks at interpolation threshold, then descends again.
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, np, plt):
    # Double Descent
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Generate complexity axis
    _x = np.linspace(0, 3, 300)

    # Key parameters
    _threshold = 1.0  # Interpolation threshold
    _sweet_spot = 0.5  # Where classical U-curve has minimum
    _min_error = 0.25  # Minimum error at sweet spot
    _peak_error = 0.95  # Peak error at interpolation threshold

    # Classical U-curve (continues rising forever after sweet spot)
    _classical = _min_error + 0.35 * (_x - _sweet_spot) ** 2

    # Actual behavior - build it in pieces
    _double_descent = np.zeros_like(_x)

    # Region 1: Underparameterized (x < threshold) - follows classical but rises to peak
    _under_mask = _x < _threshold
    # Classical curve as base (same as classical prediction)
    _base = _min_error + 0.35 * (_x[_under_mask] - _sweet_spot) ** 2
    # Extra rise near threshold - scaled to be zero at x=0
    _classical_at_threshold = _min_error + 0.35 * (_threshold - _sweet_spot) ** 2
    # Use a function that is 0 at x=0 and rises steeply near threshold
    _proximity = _x[_under_mask] / _threshold  # 0 at x=0, 1 at threshold
    _extra_rise = (_peak_error - _classical_at_threshold) * (_proximity ** 4)
    _double_descent[_under_mask] = _base + _extra_rise

    # Region 2: Overparameterized (x >= threshold) - second descent
    _over_mask = _x >= _threshold
    # Exponential decay from peak to a floor
    _floor_error = 0.15
    _double_descent[_over_mask] = (
        _floor_error + (_peak_error - _floor_error) * np.exp(-2.5 * (_x[_over_mask] - _threshold))
    )

    # Plot classical prediction (dashed, extended)
    _ax.plot(
        _x,
        _classical,
        color=COLORS["neutral"],
        linewidth=2,
        linestyle="--",
        label="Classical prediction",
        alpha=0.7,
    )

    # Plot actual double descent curve
    _ax.plot(
        _x,
        _double_descent,
        color=COLORS["primary"],
        linewidth=3,
        label="Actual behavior",
    )

    # Mark interpolation threshold
    _ax.axvline(
        x=_threshold,
        color=COLORS["danger"],
        linestyle=":",
        linewidth=2,
        alpha=0.8,
    )

    # Shade regions
    _ax.axvspan(0, _threshold, alpha=0.05, color=COLORS["primary"])
    _ax.axvspan(_threshold, 3, alpha=0.05, color=COLORS["success"])

    # Add region labels at bottom
    _ax.text(
        0.5,
        -0.12,
        "Underparameterized",
        transform=_ax.get_xaxis_transform(),
        fontsize=10,
        ha="center",
        color=COLORS["primary"],
    )
    _ax.text(
        2.0,
        -0.12,
        "Overparameterized",
        transform=_ax.get_xaxis_transform(),
        fontsize=10,
        ha="center",
        color=COLORS["success"],
    )

    # Interpolation threshold label
    _ax.annotate(
        "Interpolation\nthreshold",
        xy=(_threshold, 0.75),
        xytext=(_threshold + 0.3, 1.1),
        fontsize=10,
        ha="left",
        color=COLORS["danger"],
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["danger"],
            lw=1.5,
        ),
    )

    # Second descent annotation
    _ax.annotate(
        "Second\ndescent!",
        xy=(2.2, 0.22),
        xytext=(2.5, 0.5),
        fontsize=11,
        ha="center",
        color=COLORS["success"],
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["success"],
            lw=1.5,
        ),
    )

    # Style
    _ax.set_xlabel("Model Complexity", fontsize=12, fontweight="bold")
    _ax.set_ylabel("Test Error", fontsize=12, fontweight="bold")
    _ax.set_xlim(0, 3)
    _ax.set_ylim(0, 1.3)
    _ax.set_xticks([])
    _ax.legend(loc="upper right", fontsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "double_descent.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'double_descent.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Grokking
    **Target slide**: `Deep Dive - Surprising Phenomena in Modern Deep Learning.md`, lines 313-336

    Shows the grokking phenomenon: training accuracy rises quickly (memorization), but test accuracy stays flat for a long time before suddenly rising (delayed generalization).
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, np, plt):
    # Grokking
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Linear spacing for epochs
    _epochs = np.linspace(0, 30000, 500)

    # Training accuracy: rapid rise to 100%
    _train_acc = 100 * (1 - np.exp(-_epochs / 100))

    # Test accuracy: FLAT near random for long time, then SUDDEN rise
    # Key: use very sharp transition (small width) and later start
    _grok_start = 22000  # Grokking starts late
    _grok_width = 1500   # Sharp transition (was 5000 - too gradual)
    _random_level = 5    # Random guessing baseline

    # Build test accuracy: flat then sharp rise
    _test_acc = np.where(
        _epochs < _grok_start - 3 * _grok_width,
        _random_level,  # Flat at random level
        _random_level + 95 / (1 + np.exp(-(_epochs - _grok_start) / _grok_width))
    )

    # Plot curves
    _ax.plot(
        _epochs,
        _train_acc,
        color=COLORS["primary"],
        linewidth=2.5,
        label="Training Accuracy",
    )
    _ax.plot(
        _epochs,
        _test_acc,
        color=COLORS["accent"],
        linewidth=2.5,
        label="Test Accuracy",
    )

    # Mark memorization point (when training hits ~100%)
    _mem_epoch = 500
    _ax.axvline(
        x=_mem_epoch,
        color=COLORS["dark"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    _ax.annotate(
        "Memorization\n(fast)",
        xy=(_mem_epoch, 70),
        xytext=(_mem_epoch + 1500, 60),
        fontsize=10,
        ha="left",
        color=COLORS["dark"],
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["dark"],
            lw=1.5,
        ),
    )

    # Where the flat region ends (test starts rising)
    _flat_end = _grok_start - 3 * _grok_width  # ~17500

    # Mark grokking point (where test accuracy is rising sharply)
    _ax.axvline(
        x=_grok_start,
        color=COLORS["success"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    _ax.annotate(
        "Grokking!\n(delayed)",
        xy=(_grok_start, 50),
        xytext=(_grok_start + 2000, 70),
        fontsize=11,
        ha="left",
        color=COLORS["success"],
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["success"],
            lw=1.5,
        ),
    )

    # Add gap annotation - arrow ends where test curve actually stays flat
    _ax.annotate(
        "",
        xy=(_mem_epoch + 500, 12),
        xytext=(_flat_end, 12),
        arrowprops=dict(
            arrowstyle="<->",
            color=COLORS["dark"],
            lw=1.5,
        ),
    )
    _ax.text(
        (_mem_epoch + _flat_end) / 2,
        18,
        "Long gap (test stays at random)",
        fontsize=9,
        ha="center",
        color=COLORS["dark"],
    )

    # Style
    _ax.set_xlabel("Epochs", fontsize=12, fontweight="bold")
    _ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    _ax.set_xlim(0, 30000)
    _ax.set_ylim(0, 105)
    _ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
    _ax.set_xticklabels(["0", "5K", "10K", "15K", "20K", "25K", "30K"])
    _ax.legend(loc="center right", fontsize=10)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "grokking.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'grokking.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Emergent Abilities
    **Target slide**: `Deep Dive - Surprising Phenomena in Modern Deep Learning.md`, lines 472-490

    Shows how task performance stays flat near random across many orders of magnitude of model scale, then suddenly jumps at a threshold.
    """)
    return


@app.cell
def _(COLORS, DPI, FIG_HEIGHT, FIG_WIDTH, OUTPUT_DIR, np, plt):
    # Emergent Abilities
    _fig, _ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Log scale for parameters (1M to 1T)
    _params = np.logspace(6, 12, 300)

    # Performance: flat near 0%, then sudden S-curve jump
    # Threshold at 10^10 = 10B parameters
    _threshold_log = 10.0  # 10B parameters
    _steepness = 5  # Sharper transition
    _performance = 100 / (1 + np.exp(-(np.log10(_params) - _threshold_log) * _steepness))

    # Plot curve (no noise - clean line)
    _ax.semilogx(
        _params,
        _performance,
        color=COLORS["primary"],
        linewidth=3,
    )

    # Shade "random guessing" region
    _ax.axhspan(0, 10, alpha=0.1, color=COLORS["neutral"])
    _ax.text(
        3e7,
        5,
        "Random guessing",
        fontsize=10,
        color=COLORS["dark"],  # Darker color for better contrast
        va="center",
    )

    # Mark emergence threshold
    _threshold_val = 10 ** _threshold_log
    _ax.axvline(
        x=_threshold_val,
        color=COLORS["accent"],
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )
    _ax.annotate(
        "Emergence\nthreshold",
        xy=(_threshold_val, 50),
        xytext=(_threshold_val * 4, 35),
        fontsize=11,
        ha="left",
        color=COLORS["accent"],
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["accent"],
            lw=1.5,
        ),
    )

    # Style
    _ax.set_xlabel("Model Parameters (log scale)", fontsize=12, fontweight="bold")
    _ax.set_ylabel("Task Performance (%)", fontsize=12, fontweight="bold")
    _ax.set_xlim(1e6, 1e12)
    _ax.set_ylim(0, 105)

    # Custom x-tick labels
    _ax.set_xticks([1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
    _ax.set_xticklabels(["1M", "10M", "100M", "1B", "10B", "100B", "1T"])

    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)

    _fig.tight_layout()
    _fig.savefig(
        OUTPUT_DIR / "deep_dive" / "emergent_abilities.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()
    plt.close(_fig)
    print(f"Saved: {OUTPUT_DIR / 'deep_dive' / 'emergent_abilities.png'}")
    return


@app.cell(hide_code=True)
def _(OUTPUT_DIR, mo):
    mo.md(f"""
    # Generation Complete!

    All figures have been saved to: `{OUTPUT_DIR}`

    ## Files Generated:
    **Module 1:**
    - `module1/ml_hierarchy.png`
    - `module1/ml_task_categories.png`
    - `module1/confusion_matrix.png`
    - `module1/kfold_cv.png`
    - `module1/overfitting_underfitting.png`

    **Module 6-8:**
    - `module6/mlp_architecture.png`
    - `module7/cnn_pipeline.png`
    - `module8/rnn_unrolled.png`

    **Deep Dives:**
    - `deep_dive/sigmoid_curves.png`
    - `deep_dive/neuron_approximation.png`
    - `deep_dive/relu_function.png`
    - `deep_dive/relu_bump.png`
    - `deep_dive/xor_decision_boundary.png`
    - `deep_dive/model_spectrum.png`
    - `deep_dive/local_connectivity.png`
    - `deep_dive/max_pooling.png`
    - `deep_dive/attention_flow-1.png`
    - `deep_dive/attention_flow-2.png`
    - `deep_dive/bias_variance_u_curve.png`
    - `deep_dive/early_stopping.png`
    - `deep_dive/double_descent.png`
    - `deep_dive/grokking.png`
    - `deep_dive/emergent_abilities.png`

    ## Next Steps:
    Update each slide file to replace ASCII art with image references:
    ```markdown
    ![Description](figures/module1/filename.png)
    ```
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
