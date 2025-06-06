import random
import pandas as pd
from tkinter import ttk
import tkinter as tk

# Parâmetros
alpha = 0.5  # taxa de aprendizado # Dinamico?
gamma = 0.9  # fator de desconto # Dinamico?
epsilon = 0.8179788  # chance de exploração # Dinamico?

# Estados e ações
estados = ['muito_seco', 'seco', 'ideal', 'encharcado']
acoes = ['regar', 'pouca_agua', 'nao_regar', 'regar_bastante']

# Inicializa Q-table
q_table = {estado: {acao: 0.0 for acao in acoes} for estado in estados}

# Função de transição
def transicao(estado, acao):
    if estado == 'muito_seco':
        if acao == 'regar_bastante':
            return 'ideal'
        elif acao == 'pouca_agua':
            return 'muito_seco'
        else:
            return 'muito_seco'
    
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
        if acao == 'regar':
            return 'encharcado'
        elif acao == 'pouca_agua':
            return 'ideal'
        else:
            return 'ideal'

# Função de recompensa
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

# Histórico
historico = []

# Simulação inicial para popular dados
for episodio in range(1, 301):
    estado = random.choice(estados)
    for passo in range(1):
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

        estado = prox_estado

# Preparar dados para interface
# Dados Dinamicos podem ser definidos antes de inciar a tabela principal?

historico_df = pd.DataFrame(historico)
ultimas_linhas = historico_df.tail(10)
dados = ultimas_linhas.values.tolist()
cabecalhos = ultimas_linhas.columns.tolist()
q_table_df = pd.DataFrame(q_table).T.round(2)

# Função para reexecutar código ao clicar no botão
def botãoExecutar():
    global q_table, historico, tabela_q, tabela_hist

    q_table = {estado: {acao: 0.0 for acao in acoes} for estado in estados}
    historico = []

    for episodio in range(1, 201):
        estado = random.choice(estados)
        for passo in range(1):
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

            estado = prox_estado

    # Atualizar DataFrames
    ultimas_linhas = pd.DataFrame(historico).tail(10)
    dados = ultimas_linhas.values.tolist()
    q_table_df = pd.DataFrame(q_table).T.round(2)

    # Atualizar Treeview Q-table
    for item in tabela_q.get_children():
        tabela_q.delete(item)
    for idx, row in q_table_df.iterrows():
        tabela_q.insert("", "end", values=list(row))

    # Atualizar Treeview Histórico
    for item in tabela_hist.get_children():
        tabela_hist.delete(item)
    for linha in dados:
        tabela_hist.insert("", "end", values=linha)

# --- Interface Tkinter ---

janela = tk.Tk()
janela.title("Aprendizado por Reforço - Q-Learning")
janela.geometry("950x600")
janela.configure(bg="#000")

# Estilo Treeview
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



# Título
titulo_frame = tk.Frame(janela, bg="#000")
titulo_frame.pack(pady=10)

tk.Label(titulo_frame, text=" X ", font=("Arial Bold", 18, "bold"), bg="#000", fg="#FF0000").pack(side="left")
tk.Label(titulo_frame, text="Fork na ", font=("Arial", 18, "bold"), bg='#000', fg="#fff").pack(side="left")
tk.Label(titulo_frame, text="main", font=("Arial Bold", 18, "bold"), bg="#FF9900", fg="#000").pack(side="left")

# Frame da Tabela Q
frame_q = tk.LabelFrame(janela, text="Tabela Q Final", font=("Arial", 12, "bold"),
                        bg="#000", fg="#fff", padx=10, pady=10)
frame_q.pack(fill="both", expand=True, padx=20, pady=10)

tabela_q = ttk.Treeview(frame_q, columns=q_table_df.columns.tolist(), show="headings", height=4)
for col in q_table_df.columns:
    tabela_q.heading(col, text=col)
    tabela_q.column(col, anchor="center", width=120)

for idx, row in q_table_df.iterrows():
    tabela_q.insert("", "end", values=list(row), tags=(idx,))

tabela_q.pack(fill="both", expand=True)

# Frame do Histórico
frame_hist = tk.LabelFrame(janela, text="Últimas 10 Decisões", font=("Arial", 12, "bold"),
                           bg="#000", fg="#fff", padx=10, pady=10)
frame_hist.pack(fill="both", expand=True, padx=20, pady=10)

tabela_hist = ttk.Treeview(frame_hist, columns=cabecalhos, show="headings", height=10)
for col in cabecalhos:
    tabela_hist.heading(col, text=col)
    tabela_hist.column(col, anchor="center", width=110)

for linha in dados:
    tabela_hist.insert("", "end", values=linha)

tabela_hist.pack(fill="both", expand=True)

# Botão para reexecutar
botao_reexecutar = tk.Button(
    janela,
    text="\u25B6",            
    font=("Arial", 14, "bold"),          
    bg="#F77103",
    fg="#fff",
    activebackground="#FF9900",
    padx=12,                            
    pady=6,
    command=botãoExecutar
)
botao_reexecutar.pack(pady=15)


janela.mainloop()