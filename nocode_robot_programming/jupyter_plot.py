from IPython.display import display, HTML
import matplotlib.pyplot as plt
import io
import base64

class JupyterPlot():
    """ Plot in Jupyter notebook. Plots aligned next to each other
    Usage:
        plt.plot(plot_data)
        self.save() # this saves the fig so another can be created    
        plt.plot(second_plot_data)
        self.save()
        plt.plot(third_plot_data)
        self.save()    
        self.show() # flushes the saved figures on a screen and visualize them
    """
    def __init__(self):
        self.plots = []
    def save(self):
        self.plt_save()
    def plt_save(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        self.plots.append(img_base64)

    
    def show(self):
        html = ""
        n = len(self.plots)
        for plot in self.plots:
            html += f'<img src="data:image/png;base64,{plot}" style="width:{min(35,(100 // n) - 1)}%;display:inline-block;margin:5px;">'
        display(HTML(html))
        self.plots = []

jupyter_plot = JupyterPlot()


# Inline playback in Jupyter using Matplotlib (no ffmpeg/GIF)
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display

def show_gray_video_cuda(frames_cuda: torch.Tensor, fps: int = 20, repeat: bool = True, scale: float = 1.0):
    """
    Play a grayscale video tensor [T, H, W] (float32, 0..1) that lives on CUDA.
    Renders inline using JS (no external encoders).
    """
    assert frames_cuda.ndim == 3, "Expected [T, H, W] grayscale tensor"
    # Move to CPU and convert to uint8 for fast imshow
    frames_u8 = (frames_cuda.detach().clamp(0,1) * 255).round() \
                .to(torch.uint8).cpu().numpy()  # shape: [T, H, W]

    T, H, W = frames_u8.shape
    # fig, ax = plt.subplots()
    dpi = 100  # any dpi; we size the figure to match pixels
    fig, ax = plt.subplots(
        figsize=(scale * W / dpi, scale * H / dpi),  # ← scale applied here
        # figsize=(W / dpi, H / dpi), dpi=dpi, frameon=False,  # no figure frame
        constrained_layout=False
    )
    fig.patch.set_alpha(0)  # transparent figure background
    ax.set_position([0, 0, 1, 1])  # axes fills entire figure (no margins)
    ax.set_axis_off()
    im = ax.imshow(frames_u8[0], cmap='gray', animated=True, vmin=0, vmax=255)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    def _update(i):
        im.set_array(frames_u8[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=1000/fps, blit=True, repeat=repeat
    )
    plt.close(fig)  # avoid double display
    return HTML(f'<div style="margin:0;padding:0;line-height:0">{anim.to_jshtml()}</div>')
