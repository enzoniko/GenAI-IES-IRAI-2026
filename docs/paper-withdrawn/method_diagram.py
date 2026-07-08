import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Helper function for boxes
    def draw_box(x, y, w, h, label, color, text_size=10):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=2, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontweight='bold', fontsize=text_size, wrap=True)

    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Phase 0
    draw_box(0.5, 7.5, 2, 1.5, "Healthy Data", "#e1f5fe")
    draw_arrow(2.5, 8.25, 3.5, 8.25)
    draw_box(3.5, 7.5, 2, 1.5, "PINN\n(Physics Residuals)", "#bbdefb")
    draw_arrow(5.5, 8.25, 6.5, 8.25)
    draw_box(6.5, 7.5, 3, 1.5, "Physics-Informed\nEmbedding Space", "#90caf9")
    ax.text(0.5, 9.2, "Phase 0: Physical Foundation", fontweight='bold', fontsize=12)

    # Phase 1
    draw_box(0.5, 4.5, 2, 1.5, "Raw Signal", "#f1f8e9")
    draw_arrow(2.5, 5.25, 3.5, 5.25)
    draw_box(3.5, 4.5, 2, 1.5, "TS-JEPA\nEncoder", "#dcedc8")
    draw_arrow(5.5, 5.25, 6.5, 5.25)
    draw_box(6.5, 5.5, 2, 1, "Decoder 1\n(Envelope)", "#c5e1a5")
    draw_box(6.5, 4, 2, 1, "Decoder 2 CVAE\n(Jitter)", "#c5e1a5")
    ax.text(0.5, 6.2, "Phase 1: Representation Learning", fontweight='bold', fontsize=12)

    # Phase 2
    draw_box(0.5, 1, 1.5, 1.5, "Healthy\nz_macro", "#fff3e0")
    draw_arrow(2, 1.75, 3, 1.75)
    draw_box(3, 1, 2.5, 1.5, "SDEdit Loop\n(LDM Denoising)", "#ffe0b2")
    draw_arrow(5.5, 1.75, 6.5, 1.75)
    draw_box(6.5, 1, 2, 1.5, "Counterfactual\nSignal", "#ffcc80")
    
    # Guidance link
    draw_arrow(8, 7.5, 4.25, 2.5)
    ax.text(6, 4.5, "VJP Guidance\nfrom PINN Oracle", ha='center', va='center', fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray"))
    
    ax.text(0.5, 2.7, "Phase 2: Counterfactual Synthesis", fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig('method_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_diagram()
