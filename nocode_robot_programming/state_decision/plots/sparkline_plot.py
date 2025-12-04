
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def likelihood_sparklines_withtruelabels(
    likelihoods,             # shape: (c, n)
    class_names,             # list/seq length c
    true_labels=None,        # len n; ints in [0..c-1] or class-name strings
    truth_style="barcode",   # 'barcode' (default), 'strip', 'overlay', 'both'
    cols=6,
    sharey=True,
    figsize=None,
    dpi=200,
    fig_bg="#f6f7f9",
    tile_bg="#ffffff",
    rounding=10,
    fill_alpha=0.25,
    line_lw=1.6,
    truth_band_height=0.08,     # height for inline barcode (axes fraction)
    truth_bar_alpha=0.85,       # opacity for inline barcode
    truth_overlay_alpha=0.10,   # opacity for vertical overlay spans
):
    """
    Adds ground-truth encoding without clutter:
      - 'barcode': thin colored band at the bottom of each tile where that class is true
      - 'strip': one consolidated strip across the figure showing the true class timeline
      - 'overlay': faint vertical spans in tiles where that class is true
      - 'both': barcode + strip

    Returns (fig, axes, truth_ax_or_None)
    """
    L = np.asarray(likelihoods, dtype=float)
    if L.ndim != 2:
        raise ValueError("likelihoods must be a 2D array of shape (c, n)")
    C, N = L.shape
    if len(class_names) != C:
        raise ValueError("class_names length must match number of classes (c)")

    # Map true_labels to integer indices (0..C-1)
    true_idx = None
    if true_labels is not None:
        if len(true_labels) != N:
            raise ValueError("true_labels length must match number of steps (n)")
        name_to_idx = {str(nm): i for i, nm in enumerate(class_names)}
        if isinstance(true_labels[0], str):
            true_idx = np.array([name_to_idx[str(t)] for t in true_labels], dtype=int)
        else:
            true_idx = np.asarray(true_labels, dtype=int)
        if np.any((true_idx < 0) | (true_idx >= C)):
            raise ValueError("true_labels indices out of range")

    # Layout: add a short row for the figure-level truth strip if needed
    want_strip = truth_style in ("strip", "both")
    rows = int(np.ceil(C / cols))
    if figsize is None:
        figsize = (max(2, cols) * 1.15, (rows + (0.35 if want_strip else 0)) * 1.15)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor(fig_bg)

    if want_strip:
        gs = fig.add_gridspec(rows + 1, cols, wspace=0.1, hspace=0.2,
                              height_ratios=[1]*rows + [0.25])
        strip_row = rows
    else:
        gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.2)

    # Shared y-limits
    if sharey:
        y_min = np.nanmin(L)
        y_max = np.nanmax(L)
        if not np.isfinite(y_min): y_min = 0.0
        if not np.isfinite(y_max): y_max = 1.0
        pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        shared_ylim = (y_min - pad, y_max + pad)
    else:
        shared_ylim = None

    xs = np.linspace(0, 1, N)

    # Stable class colors (same color everywhere for a given class)
    cmap = plt.get_cmap("tab20")
    class_colors = [cmap((i*2) % 20) for i in range(C)]  # 0,2,4,... -> dark set



    axes = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        if want_strip and r == strip_row:
            # This row is reserved for the truth strip
            continue
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        if i >= C:
            ax.axis("off")
            continue

        # Rounded tile background
        ax.set_facecolor(tile_bg)
        tile = FancyBboxPatch(
            (-0.03, -0.03), 1.06, 1.06,
            boxstyle=f"round,pad=0.012,rounding_size={rounding}",
            transform=ax.transAxes, linewidth=0, facecolor=tile_bg, zorder=0
        )
        ax.add_artist(tile)

        y = L[i]
        color = class_colors[i]

        # Optional overlay: faint vertical spans where this class is the ground truth
        if true_idx is not None and truth_style in ("overlay", "both"):
            mask = (true_idx == i)
            # Build contiguous spans in data space
            def spans_from_mask(mask, xs):
                spans = []
                in_run = False
                for k, m in enumerate(mask):
                    if m and not in_run:
                        start = xs[k-1] if k > 0 else xs[0]
                        in_run = True
                    elif not m and in_run:
                        end = xs[k]
                        spans.append((start, end))
                        in_run = False
                if in_run:
                    spans.append((start, xs[-1]))
                return spans
            for s, e in spans_from_mask(mask, xs):
                ax.axvspan(s, e, color=color, alpha=truth_overlay_alpha, zorder=0.5)

        # Likelihood curve and fill
        ax.plot(xs, y, lw=line_lw, color=color, zorder=2)
        ax.fill_between(xs, 0, y, alpha=fill_alpha, color=color, zorder=1)
        ax.scatter([xs[-1]], [y[-1]], s=10, color=color, zorder=3)

        # Inline barcode band (per-tile)
        if true_idx is not None and truth_style in ("barcode", "both"):
            band_ax = ax
            band = (true_idx == i).astype(int)[None, :]  # shape (1, N)
            band_cmap = ListedColormap([(0, 0, 0, 0), color])  # 0 transparent, 1 colored
            # Draw in x-data coords and y as axes fraction (0=bottom, 1=top)
            band_ax.imshow(
                band,
                extent=(xs[0], xs[-1], 0.0, truth_band_height),
                cmap=band_cmap,
                interpolation="nearest",
                aspect="auto",
                origin="lower",
                transform=band_ax.get_xaxis_transform(),
                alpha=truth_bar_alpha,
                zorder=2.5,
            )

        # Clean micro-axes
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Y-limits
        if shared_ylim:
            ax.set_ylim(shared_ylim)
        else:
            yymin, yymax = np.nanmin(y), np.nanmax(y)
            pad = 0.05 * (yymax - yymin if yymax > yymin else 1.0)
            ax.set_ylim(yymin - pad, yymax + pad)

        def _safe_label(s, max_len=28):
            t = str(s)
            if mpl.rcParams.get('text.usetex', False):
                t = t.replace('_', r'\_')   # escape for TeX
            if len(t) > max_len:
                t = t[:max_len-1] + '…'     # keep tiles tidy
            return t

        ax.text(0.02, 0.96, _safe_label(class_names[i]),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6, weight="bold", clip_on=True)


    truth_ax = None
    if want_strip:
        # One consolidated strip across the bottom row
        truth_ax = fig.add_subplot(gs[strip_row, :])
        truth_ax.set_facecolor(tile_bg)
        if true_idx is None:
            # Empty but still visually coherent
            truth_ax.axis("off")
        else:
            strip = true_idx[None, :]  # shape (1, N)
            strip_cmap = ListedColormap(class_colors)
            truth_ax.imshow(
                strip,
                extent=(xs[0], xs[-1], 0, 1),
                cmap=strip_cmap,
                interpolation="nearest",
                aspect="auto",
                origin="lower",
            )
            # Minimal label to signify ground truth
            truth_ax.text(-0.01, 0.5, "Truth",
                          transform=truth_ax.transAxes,
                          ha="right", va="center",
                          fontsize=9, weight="bold")
            truth_ax.set_xticks([]); truth_ax.set_yticks([])
            for sp in truth_ax.spines.values():
                sp.set_visible(False)

    return fig, axes, truth_ax


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def likelihood_sparklines(
    likelihoods,             # shape: (c, n)
    class_names,             # list/seq length c
    cols=6,                  # how many tiles per row
    sharey=True,             # share y-scale across classes
    figsize=None,            # (w,h) in inches; auto if None
    dpi=200,                 # crisp small plots
    fig_bg="#f6f7f9",        # figure background color
    tile_bg="#ffffff",       # tile background
    rounding=10,             # corner radius for tiles
    fill_alpha=0.25,         # area fill opacity
    line_lw=1.6             # line thickness
):
    """
    Draws a grid of tiny likelihood plots (one per class) with minimal chrome.
    likelihoods: array-like (c, n)
    class_names: sequence of length c
    Returns (fig, axes) for further customization/saving.
    """
    L = np.asarray(likelihoods, dtype=float)
    if L.ndim != 2:
        raise ValueError("likelihoods must be a 2D array of shape (c, n)")
    C, N = L.shape
    if len(class_names) != C:
        raise ValueError("class_names length must match number of classes (c)")

    rows = int(np.ceil(C / cols))
    if figsize is None:
        # Small default: ~1.15" per tile in width/height
        figsize = (max(2, cols) * 1.15, max(1, rows) * 1.15)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor(fig_bg)
    gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.2)

    # Shared y-limits for comparability if requested
    if sharey:
        y_min = np.nanmin(L)
        y_max = np.nanmax(L)
        if not np.isfinite(y_min): y_min = 0.0
        if not np.isfinite(y_max): y_max = 1.0
        pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        shared_ylim = (y_min - pad, y_max + pad)
    else:
        shared_ylim = None

    xs = np.linspace(0, 1, N)
    cmap = plt.get_cmap("tab10")

    axes = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        if i >= C:
            ax.axis("off")
            continue

        # Rounded tile background
        ax.set_facecolor(tile_bg)
        tile = FancyBboxPatch(
            (-0.03, -0.03), 1.06, 1.06,
            boxstyle=f"round,pad=0.012,rounding_size={rounding}",
            transform=ax.transAxes, linewidth=0, facecolor=tile_bg, zorder=0
        )
        ax.add_artist(tile)

        y = L[i]
        color = cmap(i % 10)

        # Plot and fill
        line, = ax.plot(xs, y, lw=line_lw, color=color, zorder=2)
        ax.fill_between(xs, 0, y, alpha=fill_alpha, color=color, zorder=1)

        # Optional final-point marker (subtle)
        ax.scatter([xs[-1]], [y[-1]], s=10, color=color, zorder=3)

        # Remove all ticks/labels/spines for a clean micro-plot
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Y limits
        if shared_ylim:
            ax.set_ylim(shared_ylim)
        else:
            yymin, yymax = np.nanmin(y), np.nanmax(y)
            pad = 0.05 * (yymax - yymin if yymax > yymin else 1.0)
            ax.set_ylim(yymin - pad, yymax + pad)

        # Class label inside the tile
        ax.text(0.02, 0.96, str(class_names[i]),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, weight="bold")

    return fig, axes


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl

def _safe_label(s, max_len=40):
    t = str(s)
    if mpl.rcParams.get('text.usetex', False):
        t = t.replace('_', r'\_')
    return t if len(t) <= max_len else t[:max_len-1] + '…'

def likelihood_single_axis(
    likelihoods,             # shape: (c, n)
    class_names,             # list length c
    true_labels=None,        # len n; class indices [0..c-1] or names
    mode="barcode",          # 'barcode' (adds truth strip), or 'none'
    figsize=(7, 2.0),
    dpi=220,
    fig_bg="#f6f7f9",
    bg_colors=("#ffffff", "#f1f3f6"),   # alternating background band colors
    show_separators=True,    # thin lines along ranked boundaries
    separator_color="#e3e7ee",
    separator_lw=0.6,
    line_lw=1.6,
    winner_fill=True,        # fill only when class is argmax to avoid mixing
    winner_fill_alpha=0.28,
    event_x=None,                 # timestep to mark; int index (default) or float in x
    event_units="index",          # 'index' -> treat event_x as step index, 'x' -> already in [0..1] data coords
    event_line_kwargs=None,       # style overrides for the vertical line
    event_star_kwargs=None,       # style overrides for the star marker
    ):
    """
    Single-axes plot:
      • Background is painted by the zones between ranked likelihoods at each timestep.
      • Each class curve is plotted on top with stable colors.
      • Optional fill only where the class is top-1 (no color mixing).
      • Optional bottom barcode indicating ground-truth over time.

    Returns (fig, ax).
    """
    L = np.asarray(likelihoods, dtype=float)
    if L.ndim != 2:
        raise ValueError("likelihoods must be (c, n)")
    C, N = L.shape
    if len(class_names) != C:
        raise ValueError("class_names length must match c")
    xs = np.linspace(0, 1, N)

    def _event_positions(event_x, units, N):
        if event_x is None:
            return []
        # allow scalar or iterable
        xs_raw = np.atleast_1d(event_x)
        if units == "index":
            # map integer step k -> normalized x in [0,1]
            if N <= 1:
                return [0.0] * len(xs_raw)
            return [float(int(k)) / (N - 1) for k in xs_raw]
        elif units == "x":
            return [float(x) for x in xs_raw]
        else:
            raise ValueError("event_units must be 'index' or 'x'")

    # Stable class colors (dark half of tab20)
    tab20 = plt.get_cmap("tab20")
    class_colors = [tab20.colors[(i*2) % 20] for i in range(C)]

    # Map true_labels to indices (if provided)
    true_idx = None
    if true_labels is not None:
        if len(true_labels) != N:
            raise ValueError("true_labels length must match n")
        name_to_idx = {str(nm): i for i, nm in enumerate(class_names)}
        if isinstance(true_labels[0], str):
            true_idx = np.array([name_to_idx[str(t)] for t in true_labels], dtype=int)
        else:
            true_idx = np.asarray(true_labels, dtype=int)
        if np.any((true_idx < 0) | (true_idx >= C)):
            raise ValueError("true_labels indices out of range")

    # --- Background: zones between ranked likelihoods ---
    # Sort likelihoods per timestep ascending; shape: (C, N)
    order = np.argsort(L, axis=0)
    sorted_vals = np.take_along_axis(L, order, axis=0)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor(fig_bg)
    ax = fig.add_subplot(111)
    ax.set_xlim(xs[0], xs[-1])

    # Y-limits padded a touch
    ymin, ymax = float(np.nanmin(L)), float(np.nanmax(L))
    pad = 0.04 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    # Paint alternating bands between ranks: [sorted[k], sorted[k+1]]
    for k in range(C - 1):
        y0 = sorted_vals[k]
        y1 = sorted_vals[k + 1]
        ax.fill_between(
            xs, y0, y1,
            facecolor=bg_colors[k % 2],
            edgecolor='none',
            zorder=0
        )
        if show_separators:
            ax.plot(xs, y0, color=separator_color, lw=separator_lw, zorder=0.5)
    if show_separators and C > 0:
        # top envelope separator for completeness
        ax.plot(xs, sorted_vals[-1], color=separator_color, lw=separator_lw, zorder=0.5)

    # --- Foreground: class curves (and optional fills only when top-1) ---
    argmax_idx = np.argmax(L, axis=0)
    for i in range(C):
        y = L[i]
        col = class_colors[i]

        if winner_fill:
            # Fill only where this class is the maximum to avoid color overlap
            mask = (argmax_idx == i)
            # draw contiguous spans for the mask
            start = None
            for k in range(N):
                if mask[k] and start is None:
                    start = k
                if (start is not None) and (k == N-1 or not mask[k+1]):
                    end = k
                    seg_x = xs[start:end+1]
                    seg_y = y[start:end+1]
                    ax.fill_between(seg_x, [ax.get_ylim()[0]]*len(seg_x), seg_y,
                                    color=col, alpha=winner_fill_alpha, zorder=1.5)
                    start = None

        # Line on top
        ax.plot(xs, y, color=col, lw=line_lw, zorder=2.0, solid_capstyle="round")

    # Minimal: remove ticks/box
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Class labels as a compact legend-like text block (left, inside)
    from matplotlib.lines import Line2D

    # Create a custom legend using Line2D handles for accurate colors
    handles = [
        Line2D([0], [0], color=class_colors[i], lw=2, label=_safe_label(class_names[i]))
        for i in range(C)
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=min(C, 6),   # wrap long legends into rows
        fontsize=8.5,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    # Optional: bottom barcode for ground truth
    if true_idx is not None and mode == "barcode":
        strip_h = 0.08  # fraction of axes height
        strip_cmap = ListedColormap(class_colors)
        # Discrete mapping: -0.5..0.5 -> color 0, 0.5..1.5 -> color 1, ..., C-1
        norm = BoundaryNorm(np.arange(-0.5, C + 0.5, 1), C)

        y0, y1 = ax.get_ylim()
        ax.imshow(
            true_idx[None, :],
            extent=(xs[0], xs[-1], y0, y0 + strip_h * (y1 - y0)),
            cmap=strip_cmap,
            norm=norm,                 # <-- key line
            interpolation="nearest",
            aspect="auto",
            origin="lower",
            zorder=2.2
        )
        ax.text(-0.01, 0.02, "Truth", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, weight="bold", color="#2b2f36")

    # --- Optional event marker(s): vertical dashed line + orange star at bottom ---
    evt_xs = _event_positions(event_x, event_units, N)
    if evt_xs:
        # defaults
        _line_kw = dict(color="black", lw=1.2, linestyle=(0, (4, 4)), zorder=3.0)
        _star_kw = dict(marker="*", markersize=9, color="orange", zorder=3.2, clip_on=False)
        if event_line_kwargs:
            _line_kw.update(event_line_kwargs)
        if event_star_kwargs:
            _star_kw.update(event_star_kwargs)

        y0, y1 = ax.get_ylim()
        # place star a bit above the very bottom (and above the truth strip if present)
        star_y_axes = 0.015  # axes-fraction from bottom
        if (true_idx is not None) and (mode == "barcode"):
            star_y_axes = 0.03  # sit just above the barcode strip

        for x_ev in evt_xs:
            ax.axvline(x_ev, **_line_kw)
            # star anchored in x-data, y in axes coords
            ax.plot([x_ev], [star_y_axes], transform=ax.get_xaxis_transform(), **_star_kw)