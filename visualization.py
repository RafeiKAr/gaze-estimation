
# =================== Heatmap =====================================

# run: python -m visualization --input results/baseline_predictions.csv

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Cursor-centered Gaussian spatial error heatmap."
    )
    # parser.add_argument("--gt", type=str, required=True, help="Path to Ground Truth CSV (x, y)")
    # parser.add_argument("--pred", type=str, required=True, help="Path to Predictions CSV (x, y)")

    parser.add_argument("--input", type=str, required=True,
        help="Path to CSV containing gt_x, gt_y, pred_x, pred_y"
)

    parser.add_argument("--radius", type=float, default=0.10, help="Selection radius around cursor in (0,1) space")
    parser.add_argument("--sigma", type=float, default=0.20, help="Standard deviation (spread) of the Gaussian kernel")

    args = parser.parse_args()

    # 1. Load CSVs
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    df.columns = df.columns.str.strip().str.lower()

    required_columns = {'gt_x', 'gt_y', 'pred_x', 'pred_y'}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV file must contain the following columns: "
            "'gt_x', 'gt_y', 'pred_x', 'pred_y'."
        )

    gt_coords = df[['gt_x', 'gt_y']].to_numpy()
    pred_coords = df[['pred_x', 'pred_y']].to_numpy()

    
    # Calculate paired Euclidean distance errors
    point_errors = np.linalg.norm(gt_coords - pred_coords, axis=1)

    # 2. Setup Figure
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title("Cursor-Centered Gaussian Error Visualizer")

    # Left Plot: Ground Truth space with selection circle
    ax_gt.scatter(gt_coords[:, 0], gt_coords[:, 1], c='blue', edgecolors='black', s=35, label='GT Points')
    highlight_gt = ax_gt.scatter([], [], c='red', edgecolors='yellow', s=70, label='Selected GT')
    
    cursor_circle = plt.Circle((0, 0), args.radius, color='red', fill=False, linestyle='--', alpha=0.7)
    ax_gt.add_patch(cursor_circle)
    cursor_circle.set_visible(False)

    ax_gt.set_xlim(0, 1)
    ax_gt.set_ylim(0, 1)
    ax_gt.set_title("Ground Truth Space (Move Cursor)")
    ax_gt.legend(loc='upper right')
    ax_gt.grid(True, linestyle='--', alpha=0.5)

    # Right Plot: Prediction Space Heatmap
    grid_res = 200
    gx = np.linspace(0, 1, grid_res)
    gy = np.linspace(0, 1, grid_res)
    GRID_X, GRID_Y = np.meshgrid(gx, gy)

    # Upper bound colorbar to max possible error in (0,1) space
    max_possible_error = np.sqrt(2)
    heatmap = ax_pred.imshow(
        np.zeros((grid_res, grid_res)),
        extent=[0, 1, 0, 1],
        origin='lower',
        cmap='inferno',
        vmin=0,
        vmax=np.max(point_errors) if len(point_errors) > 0 else max_possible_error
    )
    
    highlight_pred = ax_pred.scatter([], [], c='cyan', edgecolors='black', s=40, label='Correlated Preds')
    cbar = fig.colorbar(heatmap, ax=ax_pred)
    cbar.set_label('Mean Euclidean Error Intensity', rotation=270, labelpad=15)

    ax_pred.set_xlim(0, 1)
    ax_pred.set_ylim(0, 1)
    ax_pred.set_title("Gaussian Error Heatmap")
    ax_pred.legend(loc='upper right')

    # 3. Dynamic Cursor Selection Event
    def on_mouse_move(event):
        if event.inaxes != ax_gt:
            cursor_circle.set_visible(False)
            fig.canvas.draw_idle()
            return

        # Center of selected area (cursor position)
        center_x, center_y = event.xdata, event.ydata
        cursor_circle.set_center((center_x, center_y))
        cursor_circle.set_visible(True)

        # Find GT points inside selection radius
        dists_to_cursor = np.sqrt((gt_coords[:, 0] - center_x)**2 + (gt_coords[:, 1] - center_y)**2)
        selected_mask = dists_to_cursor <= args.radius

        if np.any(selected_mask):
            sel_gt = gt_coords[selected_mask]
            sel_pred = pred_coords[selected_mask]
            
            # Compute average error magnitude of points in this region
            mean_error = np.mean(point_errors[selected_mask])

            # Highlight selected points
            highlight_gt.set_offsets(sel_gt)
            highlight_pred.set_offsets(sel_pred)

            # Generate 2D Gaussian Kernel centered at (center_x, center_y)
            # Formula: A * exp( -((x - x0)^2 + (y - y0)^2) / (2 * sigma^2) )
            dist_sq_from_center = (GRID_X - center_x)**2 + (GRID_Y - center_y)**2
            gaussian_heatmap = mean_error * np.exp(-dist_sq_from_center / (2 * args.sigma**2))

            heatmap.set_data(gaussian_heatmap)
            ax_pred.set_title(f"Gaussian Heatmap (Center: ({center_x:.2f}, {center_y:.2f}) | Mean Error: {mean_error:.3f})")
        else:
            highlight_gt.set_offsets(np.empty((0, 2)))
            highlight_pred.set_offsets(np.empty((0, 2)))
            heatmap.set_data(np.zeros((grid_res, grid_res)))
            ax_pred.set_title("Gaussian Error Heatmap (No GT points in selection)")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# run: python -m visualization --input results/baseline_predictions.csv
# run: python -m visualization --input results/baseline_predictions_loss2.csv
# run: python -m visualization --input results/personal/predictions_10.csv
