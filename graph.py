import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from cudaviz.mandelbrot import mandelbrot

matplotlib.use('TkAgg')

colors = ['#000000', '#76b900']
cmap = LinearSegmentedColormap.from_list("black_to_nvidia_green", colors)

class GraphPage(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.title_label = tk.Label(self, text="Mandelbrot Set", font=("Helvetica", 16))
        self.title_label.pack()

        # Input fields for N and max_iter
        self.n_label = tk.Label(self, text="N:")
        self.n_label.pack()
        self.n_entry = tk.Entry(self)
        self.n_entry.insert(0, "1000")  # Default value
        self.n_entry.pack()

        self.max_iter_label = tk.Label(self, text="Max Iter:")
        self.max_iter_label.pack()
        self.max_iter_entry = tk.Entry(self)
        self.max_iter_entry.insert(0, "5000")  # Default value
        self.max_iter_entry.pack()

        # Button to replot
        self.replot_button = tk.Button(self, text="Replot", command=self.replot)
        self.replot_button.pack()

        self.pack()

    def add_mpl_figure(self, fig):
        self.mpl_canvas = FigureCanvasTkAgg(fig, self)
        self.mpl_canvas.draw()  # Updated from .show() to .draw()
        self.mpl_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.mpl_canvas, self)
        self.toolbar.update()
        self.mpl_canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def replot(self):
        # Get values from input fields
        try:
            n = int(self.n_entry.get())
            max_iter = int(self.max_iter_entry.get())
        except ValueError:
            print("Invalid input for N or Max Iter")
            return

        # Recompute and update the plot
        grid = mandelbrot(N=n, max_iter=max_iter)
        self.mpl_canvas.figure.clear()
        ax = self.mpl_canvas.figure.add_subplot(111)
        ax.imshow(grid, cmap=cmap)
        ax.spines[:].set_visible(False)
        ax.tick_params(width=0)
        ax.set_xticks([])
        ax.set_yticks([])
        self.mpl_canvas.draw()


class MPLGraph(Figure):

    def __init__(self):
        Figure.__init__(self, figsize=(5, 5), dpi=100)
        self.plot = self.add_subplot(111)
        grid = mandelbrot(N=1000, max_iter=5000)
        self.plot.imshow(grid, cmap=cmap)
        self.plot.spines[:].set_visible(False)
        self.plot.tick_params(width=0)
        self.plot.set_xticks([])
        self.plot.set_yticks([])

fig = MPLGraph()

root = tk.Tk()
graph_page = GraphPage(root)
graph_page.add_mpl_figure(fig)

root.mainloop()