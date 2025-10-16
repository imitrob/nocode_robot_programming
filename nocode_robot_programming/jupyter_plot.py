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