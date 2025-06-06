import random
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

alpha = 0.5
gamma = 0.9
epsilon = 0.8179788

estados = ['muito_seco', 'seco', 'ideal', 'encharcado']
acoes = ['regar', 'pouca_agua', 'nao_regar', 'regar_bastante']

def transicao(estado, acao):
    if estado == 'muito_seco':
        return 'ideal' if acao == 'regar_bastante' else 'muito_seco'
    elif estado == 'seco':
        if acao == 'regar':
            return 'ideal'
        elif acao == 'pouca_agua':
            return 'seco'
        else:
            return 'muito_seco'
    elif estado == 'ideal':
        if acao == 'regar':
            return 'encharcado'
        elif acao == 'pouca_agua':
            return 'ideal'
        else:
            return 'seco'
    elif estado == 'encharcado':
        if acao == 'nao_regar':
            return 'ideal'
        elif acao == 'pouca_agua':
            return 'ideal'
        else:
            return 'encharcado'

def recompensa(estado, acao):
    if estado == 'muito_seco':
        if acao == 'regar_bastante':
            return 10
        elif acao == 'pouca_agua':
            return -5
        else:
            return -10
    
    elif estado == 'seco':
        if acao == 'regar':
            return 10
        elif acao == 'pouca_agua':
            return 3
        else:
            return -10
    
    elif estado == 'ideal':
        if acao == 'nao_regar':
            return 3
        elif acao == 'pouca_agua':
            return 10
        else:
            return -8  # desperdício
    
    elif estado == 'encharcado':
        if acao == 'nao_regar':
            return 10
        elif acao == 'pouca_agua':
            return -5
        else:
            return -15  # desperdício total

def treinar_q_table(episodios, alpha, gamma, epsilon):
    q_table = {estado: {acao: 0.0 for acao in acoes} for estado in estados}
    historico = []

    for episodio in range(1, episodios + 1):
        estado = random.choice(estados)

        if random.random() < epsilon:
            acao = random.choice(acoes)
        else:
            acao = max(q_table[estado], key=q_table[estado].get)

        prox_estado = transicao(estado, acao)
        r = recompensa(estado, acao)

        q_atual = q_table[estado][acao]
        max_q_prox = max(q_table[prox_estado].values())
        q_novo = q_atual + alpha * (r + gamma * max_q_prox - q_atual)
        q_table[estado][acao] = q_novo

        historico.append({
            'Episódio': episodio,
            'Estado': estado,
            'Ação': acao,
            'Recompensa': r,
            'Próximo estado': prox_estado,
            'Q(s,a)': round(q_novo, 2)
        })

    return q_table, historico

def criar_interface():
    janela = tk.Tk()
    janela.title("Aprendizado por Reforço - Q-Learning")
    janela.geometry("950x600")
    janela.configure(bg="#000")

    alpha_var = tk.StringVar(value="0.5")
    gamma_var = tk.StringVar(value="0.9")
    epsilon_var = tk.StringVar(value="0.8")

    episodios_var = tk.StringVar(value="200")

    titulo_frame = tk.Frame(janela, bg="#000")
    titulo_frame.pack(pady=10)
    tk.Label(titulo_frame, text=" X ", font=("Arial Bold", 18), bg="#000", fg="#FF0000").pack(side="left")
    tk.Label(titulo_frame, text="Fork na ", font=("Arial", 18, "bold"), bg="#000", fg="#fff").pack(side="left")
    tk.Label(titulo_frame, text="main", font=("Arial Bold", 18), bg="#FF9900", fg="#000").pack(side="left")

    param_frame = tk.Frame(janela, bg="#000")
    param_frame.pack(pady=10)
    tk.Label(param_frame, text="Taxa de aprendizado:", bg="#000", fg="white").grid(row=0, column=0, padx=5)
    tk.Entry(param_frame, textvariable=alpha_var, width=10).grid(row=0, column=1)
    tk.Label(param_frame, text="Fator de desconto:", bg="#000", fg="white").grid(row=0, column=2, padx=5)
    tk.Entry(param_frame, textvariable=gamma_var, width=10).grid(row=0, column=3)
    tk.Label(param_frame, text="Chance de exploração:", bg="#000", fg="white").grid(row=0, column=4, padx=5)
    tk.Entry(param_frame, textvariable=epsilon_var, width=10).grid(row=0, column=5)

    tk.Label(param_frame, text="Episódios:", bg="#000", fg="white").grid(row=0, column=6, padx=5)
    tk.Entry(param_frame, textvariable=episodios_var, width=10).grid(row=0, column=7)

    frame_q = tk.LabelFrame(janela, text="Tabela Q Final", bg="#000", fg="#fff", font=("Arial", 12, "bold"), padx=10, pady=10)
    frame_q.pack(fill="both", expand=True, padx=20, pady=5)
    tabela_q = ttk.Treeview(frame_q, show="headings")
    tabela_q.pack(fill="both", expand=True)

    frame_hist = tk.LabelFrame(janela, text="Últimas 10 Decisões", bg="#000", fg="#fff", font=("Arial", 12, "bold"), padx=10, pady=10)
    frame_hist.pack(fill="both", expand=True, padx=20, pady=5)
    tabela_hist = ttk.Treeview(frame_hist, show="headings")
    tabela_hist.pack(fill="both", expand=True)

    style = ttk.Style()
    style.theme_use("default")
    style.configure("Treeview",
                    background="#D3D3D3",
                    foreground="black",
                    fieldbackground="#D3D3D3",
                    rowheight=25,
                    font=("Arial", 10))
    style.configure("Treeview.Heading",
                    background="#444",
                    foreground="white",
                    font=("Arial", 11, "bold"))

    style.map("Treeview.Heading",
            background=[("active", "#444"), ("hover", "#444")],
            foreground=[("active", "white"), ("hover", "white")])

    def executar():
        try:
            alpha_val = float(alpha_var.get())
            gamma_val = float(gamma_var.get())
            epsilon_val = float(epsilon_var.get())
            episodios_val = int(float(episodios_var.get()))
        except ValueError:
            messagebox.showerror("Erro", "Insira valores numéricos válidos.")
            return

        q_table, historico = treinar_q_table(episodios_val, alpha_val, gamma_val, epsilon_val)
        q_df = pd.DataFrame(q_table).T.round(2)
        hist_df = pd.DataFrame(historico).tail(10)

        tabela_q.delete(*tabela_q.get_children())

        colunas_q = ["Estado"] + q_df.columns.to_list()
        tabela_q["columns"] = colunas_q

        for col in colunas_q:
            tabela_q.heading(col, text=col)
            tabela_q.column(col, anchor="center", width=120)

        for idx, row in q_df.iterrows():
            tabela_q.insert("", "end", values=[idx] + list(row))

        tabela_hist.delete(*tabela_hist.get_children())
        tabela_hist["columns"] = hist_df.columns.tolist()
        for col in hist_df.columns:
            tabela_hist.heading(col, text=col)
            tabela_hist.column(col, anchor="center", width=110)
        for linha in hist_df.values.tolist():
            tabela_hist.insert("", "end", values=linha)

    botao_reexecutar = tk.Button(janela, text="\u25B6 Executar", font=("Arial", 14, "bold"), bg="#F77103", fg="#fff",
                                 activebackground="#FF9900", padx=12, pady=6, command=executar)
    botao_reexecutar.pack(pady=15)

    janela.mainloop()

if __name__ == "__main__":
    criar_interface()