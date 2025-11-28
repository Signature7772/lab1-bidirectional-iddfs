import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import time

from collections import deque

def depth_limited_dfs_collect(G, source, limit):
    visited = set()
    parent = {source: None}
    nodes = 0
    def dfs(u, depth):
        nonlocal nodes
        if depth > limit:
            return
        if u in visited:
            return
        visited.add(u)
        nodes += 1
        for v in G.neighbors(u):
            if v not in visited:
                parent[v] = u
                dfs(v, depth+1)
    dfs(source, 0)
    return visited, parent, nodes


def bidirectional_iddfs(G, start, goal, max_depth=30):
    if start == goal:
        return [start], 0, 1
    nodes_expanded = 0
    best_global_visited = 0
    for d in range(0, max_depth+1):
        vf, pf, nf = depth_limited_dfs_collect(G, start, d)
        vb, pb, nb = depth_limited_dfs_collect(G, goal, d)
        nodes_expanded += nf + nb
        best_global_visited = max(best_global_visited, len(vf) + len(vb))
        inter = vf & vb
        if inter:
            # Для надійності перевіряємо всіх кандидатів перетину і повертаємо найкоротший шлях
            best_path = None
            for meet in inter:
                path = reconstruct_path_meet(pf, pb, meet)
                if best_path is None or len(path) < len(best_path):
                    best_path = path
            return best_path, nodes_expanded, len(vf) + len(vb)
    return None, nodes_expanded, best_global_visited


def reconstruct_path_meet(pf, pb, meet):
    # pf: батьки від старту, pb: батьки від цілі
    path_f = []
    v = meet
    while v is not None:
        path_f.append(v)
        v = pf.get(v)
    path_f = list(reversed(path_f))
    path_b = []
    v = pb.get(meet)
    while v is not None:
        path_b.append(v)
        v = pb.get(v)
    return path_f + path_b

# --------- GUI / Інтерфейс ---------

class LabApp:
    def __init__(self, root):
        self.root = root
        root.title('Лабораторна робота 1 — Варіант 3: Двонаправлений сліпий пошук (IDDFS)')
        self.G = None
        self.pos = None
        self.test_results = []  # список результатів для трьох тестів: dict з keys: seed, G, pos, path, stats

        # Параметри
        params_frame = ttk.LabelFrame(root, text='Параметри графа')
        params_frame.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)

        ttk.Label(params_frame, text='Кількість вершин (n):').grid(row=0, column=0, sticky='w')
        self.n_var = tk.IntVar(value=30)
        ttk.Entry(params_frame, textvariable=self.n_var, width=6).grid(row=0, column=1, sticky='w')

        ttk.Label(params_frame, text='Ймовірність ребра (0..1):').grid(row=1, column=0, sticky='w')
        self.p_var = tk.DoubleVar(value=0.08)
        ttk.Entry(params_frame, textvariable=self.p_var, width=6).grid(row=1, column=1, sticky='w')

        ttk.Label(params_frame, text='Random seed (ціле число):').grid(row=2, column=0, sticky='w')
        self.seed_var = tk.IntVar(value=42)
        ttk.Entry(params_frame, textvariable=self.seed_var, width=8).grid(row=2, column=1, sticky='w')

        ttk.Label(params_frame, text='Початкова вершина:').grid(row=3, column=0, sticky='w')
        self.start_var = tk.IntVar(value=0)
        ttk.Entry(params_frame, textvariable=self.start_var, width=6).grid(row=3, column=1, sticky='w')

        ttk.Label(params_frame, text='Цільова вершина:').grid(row=4, column=0, sticky='w')
        self.goal_var = tk.IntVar(value=29)
        ttk.Entry(params_frame, textvariable=self.goal_var, width=6).grid(row=4, column=1, sticky='w')

        ttk.Label(params_frame, text='Макс. глибина (для IDDFS):').grid(row=5, column=0, sticky='w')
        self.maxdepth_var = tk.IntVar(value=15)
        ttk.Entry(params_frame, textvariable=self.maxdepth_var, width=6).grid(row=5, column=1, sticky='w')

        # Кнопки
        btn_frame = ttk.Frame(root)
        btn_frame.grid(row=1, column=0, sticky='ew', padx=8)
        ttk.Button(btn_frame, text='Створити граф', command=self.create_graph).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text='Генерувати 3 тестових графи', command=self.generate_three_tests).grid(row=0, column=1, padx=4)
        ttk.Button(btn_frame, text='Показати матрицю суміжності', command=self.show_adj_matrix).grid(row=0, column=2, padx=4)
        ttk.Button(btn_frame, text='Запустити пошук (IDDFS двонаправлений)', command=self.run_search).grid(row=0, column=3, padx=4)

        # Статистика / вивід
        output_frame = ttk.LabelFrame(root, text='Результати та матриця')
        output_frame.grid(row=2, column=0, sticky='nsew', padx=8, pady=8)
        self.stats_text = tk.StringVar(value='Граф не згенеровано')
        ttk.Label(output_frame, textvariable=self.stats_text).grid(row=0, column=0, sticky='w')

        self.adj_text = scrolledtext.ScrolledText(output_frame, width=50, height=12)
        self.adj_text.grid(row=1, column=0, sticky='nsew', pady=4)

        # Панель з графом
        graph_frame = ttk.LabelFrame(root, text='Візуалізація графу')
        graph_frame.grid(row=0, column=1, rowspan=3, sticky='nsew', padx=8, pady=8)
        self.figure = Figure(figsize=(6,6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Список тестів (для трьох графів)
        tests_frame = ttk.LabelFrame(root, text='Тестові графи (3)')
        tests_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=8, pady=8)
        self.tests_list = tk.Listbox(tests_frame, height=4, width=60)
        self.tests_list.grid(row=0, column=0, sticky='w')
        self.tests_list.bind('<<ListboxSelect>>', self.on_test_select)

        instr = ttk.Label(root, text='Інструкція: варіант 3 вимагає: Сліпий двонаправлений пошук в глибину на неорієнтованому графі (n=30). Натисніть "Генерувати 3 тестових графи" або створіть свій граф і "Запустити пошук".')
        instr.grid(row=4, column=0, columnspan=2, sticky='w', padx=8, pady=6)

        root.columnconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)

    def generate_graph(self, seed, set_self=False):
        # Генерує граф з заданим seed і повертає (G, pos). Якщо set_self=True — також встановлює self.G і self.pos
        n = self.n_var.get()
        if n != 30:
            n = 30
            self.n_var.set(30)
        p = self.p_var.get()
        random.seed(seed)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    G.add_edge(i, j)
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for k in range(len(comps)-1):
                a = next(iter(comps[k]))
                b = next(iter(comps[k+1]))
                G.add_edge(a, b)
        pos = nx.spring_layout(G, seed=seed)
        if set_self:
            self.G = G
            self.pos = pos
        return G, pos

    def create_graph(self, seed=None):
        if seed is None:
            seed = self.seed_var.get()
        G, pos = self.generate_graph(seed, set_self=True)
        self.stats_text.set(f'Згенеровано граф: n={self.n_var.get()}, p={self.p_var.get()}, seed={seed}, ребер={G.number_of_edges()}')
        self.draw_graph()

    def draw_graph(self, path=None, graph=None, pos=None):
        self.ax.clear()
        G = graph if graph is not None else self.G
        pos_use = pos if pos is not None else self.pos
        if G is None:
            self.ax.text(0.5,0.5,'Граф не згенеровано', ha='center')
            self.canvas.draw()
            return
        nx.draw_networkx_nodes(G, pos_use, ax=self.ax, node_size=200)
        nx.draw_networkx_labels(G, pos_use, ax=self.ax, font_size=8)
        if path is None:
            nx.draw_networkx_edges(G, pos_use, ax=self.ax)
        else:
            edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos_use, ax=self.ax)
            nx.draw_networkx_edges(G, pos_use, edgelist=edges, ax=self.ax, width=3)
            nx.draw_networkx_nodes(G, pos_use, nodelist=path, ax=self.ax, node_size=250, node_color='red')
        self.ax.set_axis_off()
        self.canvas.draw()

    def show_adj_matrix(self):
        if self.G is None:
            messagebox.showwarning('Попередження', 'Спочатку згенеруйте граф')
            return
        n = self.n_var.get()
        mat = [[0]*n for _ in range(n)]
        for u,v in self.G.edges():
            mat[u][v] = 1
            mat[v][u] = 1

        # Відкриваємо окреме вікно (Toplevel) для показу матриці — щоб не ламати основний інтерфейс
        win = tk.Toplevel(self.root)
        win.title(f'Матриця суміжності (n={n})')
        win.geometry('800x600')

        # Текстовий віджет з горизонтальною та вертикальною прокруткою
        text = tk.Text(win, wrap='none', font=('Courier', 10))
        vsb = tk.Scrollbar(win, orient='vertical', command=text.yview)
        hsb = tk.Scrollbar(win, orient='horizontal', command=text.xview)
        text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        text.pack(side='left', fill='both', expand=True)

        # Форматуємо і вставляємо матрицю з табуляцією. Використовуємо моноширинний шрифт для вирівнювання.
        header = '\t' + '\t'.join(str(i) for i in range(n)) + '\n'
        text.insert('end', header)
        for i, row in enumerate(mat):
            line = str(i) + '\t' + '\t'.join(str(x) for x in row) + '\n'
            text.insert('end', line)

        # Робимо вікно змінним за розміром
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(0, weight=1)


    def run_search(self):
        if self.G is None:
            messagebox.showwarning('Попередження', 'Спочатку згенеруйте граф')
            return
        start = self.start_var.get()
        goal = self.goal_var.get()
        n = self.n_var.get()
        if not (0 <= start < n and 0 <= goal < n):
            messagebox.showerror('Помилка', 'Вершини мають бути в діапазоні 0..n-1')
            return
        maxd = self.maxdepth_var.get()
        start_time = time.perf_counter()
        path, nodes_expanded, visited_count = bidirectional_iddfs(self.G, start, goal, max_depth=maxd)
        duration = time.perf_counter() - start_time
        if path is None:
            self.stats_text.set(f'Шлях не знайдено (max_depth={maxd}). Час: {duration:.4f}s, вузлів розгорнуто: {nodes_expanded}')
            self.draw_graph()
        else:
            self.stats_text.set(f'Знайдено шлях довжиною {len(path)-1} ребер. Час: {duration:.4f}s, вузлів розгорнуто: {nodes_expanded}, відвідано: {visited_count}')
            self.draw_graph(path=path)

    def generate_three_tests(self):
        # Генеруємо 3 різних графи незалежно — не перезаписуючи self.G поки не закінчимо
        seeds = [random.randint(1,10**6) for _ in range(3)]
        self.test_results.clear()
        self.tests_list.delete(0, tk.END)
        p = self.p_var.get()
        for s in seeds:
            G, pos = self.generate_graph(s, set_self=False)
            # Виконуємо пошук на копії графа
            start = self.start_var.get()
            goal = self.goal_var.get()
            maxd = self.maxdepth_var.get()
            path, nodes_expanded, visited_count = bidirectional_iddfs(G, start, goal, max_depth=maxd)
            status = 'знайдено' if path is not None else 'не знайдено'
            self.test_results.append({'seed': s, 'G': G, 'pos': pos, 'path': path, 'status': status, 'edges': G.number_of_edges(), 'nodes_expanded': nodes_expanded, 'visited': visited_count})
            self.tests_list.insert(tk.END, f'seed={s}, ребер={G.number_of_edges()}, шлях={status}')
        messagebox.showinfo('Тести згенеровано', 'Згенеровано 3 тестових графи. Оберіть один у списку, щоб переглянути деталі.')
        # покажемо перший тест
        if self.test_results:
            self.show_test(0)

    def on_test_select(self, event):
        sel = self.tests_list.curselection()
        if not sel:
            return
        idx = sel[0]
        self.show_test(idx)

    def show_test(self, idx):
        data = self.test_results[idx]
        # Не перезаписуємо глобальний граф доки користувач явно не захоче — але для зручності відобразимо тестовий граф
        self.G = data['G']
        self.pos = data['pos']
        self.draw_graph(path=data['path'], graph=self.G, pos=self.pos)
        self.stats_text.set(f'(test) seed={data["seed"]}, ребер={data["edges"]}, шлях={data["status"]}, вузлів розгорнуто={data["nodes_expanded"]}')


if __name__ == '__main__':
    root = tk.Tk()
    app = LabApp(root)
    root.mainloop()
