#!/usr/bin/env python3
# ga_gui_variant3.py
# GUI for Genetic Algorithm — Variant 3 (f(x) = -(x/256)^2 + 5x + 15)
# Author: (твій ім'я)
# Usage: python ga_gui_variant3.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import random
import math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --------------- Core GA functions ----------------

def f_variant3(x: int) -> float:
    """Fitness function for variant 3: f(x) = -(x/256)^2 + 5x + 15"""
    return -((x/256)**2) + 5*x + 15

def int_to_bin(x: int, bits: int=8) -> str:
    return format(x, '0{}b'.format(bits))

def bin_to_int(s: str) -> int:
    return int(s, 2)

def evaluate_population(pop, fitness_fn):
    vals = [fitness_fn(x) for x in pop]
    total = sum(vals)
    if total <= 0:
        fnorm = [1/len(vals)] * len(vals)
    else:
        fnorm = [v/total for v in vals]
    return vals, fnorm, total

def roulette_select(pop, fnorm):
    """Return index selected by roulette using normalized probabilities fnorm (sum to ~1)."""
    r = random.random()
    cum = 0.0
    for i,p in enumerate(fnorm):
        cum += p
        if r <= cum:
            return i
    return len(fnorm)-1

def one_point_crossover(bin_a, bin_b, bits):
    pt = random.randint(1, bits-1)
    a_new = bin_a[:pt] + bin_b[pt:]
    b_new = bin_b[:pt] + bin_a[pt:]
    return a_new, b_new, pt

def mutate_bin(bin_s, pm):
    arr = list(bin_s)
    flips = []
    for i,ch in enumerate(arr):
        if random.random() < pm:
            arr[i] = '1' if ch == '0' else '0'
            flips.append(i)
    return ''.join(arr), flips

# ------------------ GUI Application ------------------

class GAGUIApplication:
    def __init__(self, root):
        self.root = root
        root.title("GA Lab2 — Варіант 3 (Інтерфейс)")
        self.bits = 8

        # State
        self.history = []  # each element: dict with keys: pop (ints), bin (strs), fitness, fnorm
        self.current_gen = -1

        # --- Left frame: controls ---
        ctrl = ttk.LabelFrame(root, text="Параметри та керування")
        ctrl.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)

        # Parameters
        row = 0
        ttk.Label(ctrl, text="Розмір популяції (n):").grid(row=row, column=0, sticky='w')
        self.n_var = tk.IntVar(value=12)
        ttk.Entry(ctrl, textvariable=self.n_var, width=8).grid(row=row, column=1, sticky='w')
        row += 1

        ttk.Label(ctrl, text="Ймовірність схрещування (pc):").grid(row=row, column=0, sticky='w')
        self.pc_var = tk.DoubleVar(value=0.75)
        ttk.Entry(ctrl, textvariable=self.pc_var, width=8).grid(row=row, column=1, sticky='w')
        row += 1

        ttk.Label(ctrl, text="Ймовірність мутації (pm):").grid(row=row, column=0, sticky='w')
        self.pm_var = tk.DoubleVar(value=0.001)
        ttk.Entry(ctrl, textvariable=self.pm_var, width=8).grid(row=row, column=1, sticky='w')
        row += 1

        ttk.Label(ctrl, text="Поколінь за запуск:").grid(row=row, column=0, sticky='w')
        self.gens_var = tk.IntVar(value=5)
        ttk.Entry(ctrl, textvariable=self.gens_var, width=8).grid(row=row, column=1, sticky='w')
        row += 1

        ttk.Label(ctrl, text="Seed (ціле):").grid(row=row, column=0, sticky='w')
        self.seed_var = tk.IntVar(value=42)
        ttk.Entry(ctrl, textvariable=self.seed_var, width=8).grid(row=row, column=1, sticky='w')
        row += 1

        # Buttons
        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=(8,0))
        ttk.Button(btn_frame, text="Ініціалізувати популяцію",
                   command=self.init_population).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(btn_frame, text="Запустити 1 покоління",
                   command=self.run_one_generation).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(btn_frame, text="Запустити N поколінь",
                   command=self.run_n_generations).grid(row=0, column=2, padx=4, pady=2)
        row += 1

        ttk.Button(ctrl, text="Показати таблицю покоління", command=self.show_gen_table).grid(row=row, column=0, columnspan=2, sticky='ew', pady=6)
        row += 1

        ttk.Button(ctrl, text="Показати графік (best/avg)", command=self.plot_history).grid(row=row, column=0, columnspan=2, sticky='ew')
        row += 1

        # --- Middle frame: generation list + table ---
        mid = ttk.LabelFrame(root, text="Покоління / Таблиця популяції")
        mid.grid(row=0, column=1, sticky='nsew', padx=8, pady=8)
        root.columnconfigure(1, weight=1)

        # Generation selector
        ttk.Label(mid, text="Оберіть покоління:").grid(row=0, column=0, sticky='w')
        self.gen_list = ttk.Combobox(mid, values=[], state='readonly', width=20)
        self.gen_list.grid(row=0, column=1, sticky='w')
        self.gen_list.bind("<<ComboboxSelected>>", lambda e: self.display_selected_generation())

        # Treeview for table
        cols = ("idx","binary","x","f(x)","fnorm")
        self.tree = ttk.Treeview(mid, columns=cols, show='headings', height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=80, anchor='center')
        self.tree.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=4, pady=4)
        mid.rowconfigure(1, weight=1)
        mid.columnconfigure(1, weight=1)

        # --- Right frame: plot and log ---
        right = ttk.LabelFrame(root, text="Графік та лог")
        right.grid(row=0, column=2, sticky='nsew', padx=8, pady=8)
        root.columnconfigure(2, weight=1)

        # Matplotlib Figure
        self.fig = Figure(figsize=(5,4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)

        # Log window
        ttk.Label(right, text="Лог:").pack(anchor='w', padx=4)
        self.log = scrolledtext.ScrolledText(right, height=8)
        self.log.pack(fill='both', expand=False, padx=4, pady=(0,4))

        # Initialize with default (empty)
        self.log_message("Готово. Задайте параметри та натисніть 'Ініціалізувати популяцію'.")

    # ---------- Helper / UI methods ----------

    def log_message(self, msg: str):
        self.log.insert('end', msg + "\n")
        self.log.see('end')

    def update_gen_list(self):
        vals = [f"gen {i}" for i in range(len(self.history))]
        self.gen_list['values'] = vals
        if vals:
            self.gen_list.current(len(vals)-1)

    def display_selected_generation(self):
        sel = self.gen_list.current()
        if sel < 0: return
        self.show_table_for_gen(sel)

    def show_table_for_gen(self, gen_idx):
        if gen_idx < 0 or gen_idx >= len(self.history):
            messagebox.showwarning("Попередження", "Оберіть існуюче покоління.")
            return
        data = self.history[gen_idx]
        self.tree.delete(*self.tree.get_children())
        for i, (b, x, vx, fn) in enumerate(zip(data['bin'], data['pop'], data['fitness'], data['fnorm'])):
            self.tree.insert('', 'end', values=(i, b, x, f"{vx:.6f}", f"{fn:.6f}"))

    def show_gen_table(self):
        sel = self.gen_list.current()
        if sel < 0:
            if not self.history:
                messagebox.showinfo("Інфо", "Поки нема поколінь. Спочатку ініціалізуйте популяцію.")
                return
            sel = len(self.history)-1
        self.show_table_for_gen(sel)

    # ---------- Core GA actions ----------

    def init_population(self):
        try:
            seed = int(self.seed_var.get())
        except Exception:
            seed = None
        if seed is not None:
            random.seed(seed)
        n = max(2, int(self.n_var.get()))
        pop = [random.randint(0, 2**self.bits - 1) for _ in range(n)]
        bins = [int_to_bin(x, self.bits) for x in pop]
        fitness, fnorm, total = evaluate_population(pop, f_variant3)
        gen0 = {'pop': pop, 'bin': bins, 'fitness': fitness, 'fnorm': fnorm}
        self.history = [gen0]
        self.current_gen = 0
        self.update_gen_list()
        self.log_message(f"Ініціалізовано популяцію n={n}, seed={seed}. Сума fitness={total:.6f}")
        self.show_table_for_gen(0)
        self.plot_history()

    def step_generation(self):
        if not self.history:
            messagebox.showwarning("Помилка", "Спочатку ініціалізуйте популяцію.")
            return None
        pc = float(self.pc_var.get())
        pm = float(self.pm_var.get())
        bits = self.bits
        prev = self.history[-1]
        pop_prev = prev['pop']
        fitness_prev = prev['fitness']
        fnorm_prev = prev['fnorm']
        n = len(pop_prev)

        # Selection (roulette) -> form selected lists
        selected_bin = []
        for i in range(n):
            idx = roulette_select(pop_prev, fnorm_prev)
            selected_bin.append(int_to_bin(pop_prev[idx], bits))

        # Crossover (pairwise one-point)
        offspring_bin = selected_bin[:]
        i = 0
        while i < n:
            a = offspring_bin[i]
            b = offspring_bin[i+1] if i+1 < n else offspring_bin[i]
            if random.random() < pc:
                a_new, b_new, pt = one_point_crossover(a, b, bits)
                offspring_bin[i] = a_new
                if i+1 < n:
                    offspring_bin[i+1] = b_new
                self.log_message(f"Кросовер pair({i},{i+1}) point={pt}")
            i += 2

        # Mutation
        mutation_events = []
        for idx,s in enumerate(offspring_bin):
            s_new, flips = mutate_bin(s, pm)
            if flips:
                mutation_events.append((idx, flips))
            offspring_bin[idx] = s_new

        if mutation_events:
            for ev in mutation_events:
                self.log_message(f"Мутація у індексі {ev[0]} біти {ev[1]}")

        # Build numeric population
        offspring = [bin_to_int(s) for s in offspring_bin]
        fitness_new, fnorm_new, total_new = evaluate_population(offspring, f_variant3)

        gen = {'pop': offspring, 'bin': offspring_bin, 'fitness': fitness_new, 'fnorm': fnorm_new}
        self.history.append(gen)
        self.current_gen += 1
        self.update_gen_list()

        best = max(fitness_new)
        avg = sum(fitness_new) / len(fitness_new)
        self.log_message(f"Створено покоління {self.current_gen}: best={best:.6f}, avg={avg:.6f}, total={total_new:.6f}")
        return gen

    def run_one_generation(self):
        self.step_generation()
        # show latest
        self.gen_list.current(len(self.history)-1)
        self.show_table_for_gen(len(self.history)-1)
        self.plot_history()

    def run_n_generations(self):
        try:
            gens = int(self.gens_var.get())
            if gens <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Помилка", "Кількість поколінь має бути натуральним числом.")
            return
        for _ in range(gens):
            self.step_generation()
        self.gen_list.current(len(self.history)-1)
        self.show_table_for_gen(len(self.history)-1)
        self.plot_history()

    def plot_history(self):
        if not self.history:
            messagebox.showinfo("Інфо", "Поки нема даних для побудови графіка.")
            return
        gens = list(range(len(self.history)))
        bests = [max(h['fitness']) for h in self.history]
        avgs = [sum(h['fitness'])/len(h['fitness']) for h in self.history]
        self.ax.clear()
        self.ax.plot(gens, bests, marker='o', label='best')
        self.ax.plot(gens, avgs, marker='o', label='avg')
        self.ax.set_xlabel('Покоління')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Best & Average fitness per generation')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

# --------------- Main ---------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GAGUIApplication(root)
    root.geometry("1200x600")
    root.mainloop()
