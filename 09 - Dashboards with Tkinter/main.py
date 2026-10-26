import tkinter as tk
from tkinter import ttk

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SimpleDashboard:

    def __init__(self, root):
        self.root = root
        self.root.title("Simple Dashboard with Tkinter")
        self.root.geometry("1000x600")
       
        self.df = pd.read_csv('./titanic/train.csv')

        # Create a frame for the dashboard
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(control_frame, text='Column:').pack(side=tk.LEFT)
        self.column_var = tk.StringVar(value='Age')

        combo = ttk.Combobox(control_frame, textvariable=self.column_var, values=['Age', 'Fare'], width =10)
        combo.pack(side=tk.LEFT, padx=5, pady=5)
        combo.bind("<<ComboboxSelected>>", self.update_charts)

        chart_frame = tk.Frame(root)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.fig, ((self.ax1, self.ax2),(self.ax3, self.ax4)) = plt.subplots(2,2, figsize=(12,8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_charts()

    def update_charts(self, event=None):
        col = self.column_var.get()

        self.ax1.clear()
        survival = self.df['Survived'].value_counts()
        self.ax1.pie(survival.values, labels=['Not Survived', 'Survived'], autopct='%1.1f%%')
        self.ax1.set_title('Survival Rate')

        self.ax2.clear()
        self.ax2.hist(self.df[col], bins=20, color='skyblue', alpha =0.7)

        self.ax3.clear()
        classes = self.df['Pclass'].value_counts().sort_index()
        self.ax3.bar(classes.index, classes.values, color=['red', 'green', 'blue'])
        self.ax3.set_title('Passenger Class Distribution')

        self.ax4.clear()
        for survived in [0,1]:
            subset = self.df[self.df['Survived'] == survived]
            self.ax4.scatter(subset[col], subset['Fare'], label='Survived' if survived else 'Not Survived', alpha=0.6)

        self.ax4.set_xlabel(col)
        self.ax4.set_ylabel('Fare')
        self.ax4.set_title('Fare vs ' + col)
        self.ax4.legend()

        self.fig.tight_layout()

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleDashboard(root)
    root.mainloop()