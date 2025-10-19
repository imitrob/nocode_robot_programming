from IPython.display import display, HTML
import matplotlib.pyplot as plt
import io
import base64

class JupyterPlot():
    """ Plot next to each other """
    def __init__(self):
        self.html = ""
    def plt_save(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        self.html += f'<img src="data:image/png;base64,{img_base64}" style="width:30%;display:inline-block;margin:5px;">'
    
    def show(self):
        display(HTML(self.html))
        self.html = ""

jupyter_plot = JupyterPlot()


# Inline playback in Jupyter using Matplotlib (no ffmpeg/GIF)
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display

def show_gray_video_cuda(frames_cuda: torch.Tensor, fps: int = 20, repeat: bool = True):
    """
    Play a grayscale video tensor [T, H, W] (float32, 0..1) that lives on CUDA.
    Renders inline using JS (no external encoders).
    """
    assert frames_cuda.ndim == 3, "Expected [T, H, W] grayscale tensor"
    # Move to CPU and convert to uint8 for fast imshow
    frames_u8 = (frames_cuda.detach().clamp(0,1) * 255).round() \
                .to(torch.uint8).cpu().numpy()  # shape: [T, H, W]

    T, H, W = frames_u8.shape
    fig, ax = plt.subplots()
    im = ax.imshow(frames_u8[0], cmap='gray', animated=True, vmin=0, vmax=255)
    ax.set_axis_off()

    def _update(i):
        im.set_array(frames_u8[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=1000/fps, blit=True, repeat=repeat
    )
    plt.close(fig)  # avoid double display
    return HTML(anim.to_jshtml())
