"""Funções compartilhadas entre o treino e o app (garante pré-processamento idêntico)."""
import pandas as pd

COL_CAT = ["Filial", "Tipo_Cliente", "Genero", "Linha_Produto",
           "Pagamento", "Dia_Semana", "Faixa_Ticket"]
COL_NUM = ["Preco_Unitario", "Quantidade", "Total", "Avaliacao", "Hora"]
COLUNAS = COL_CAT + COL_NUM


def adicionar_faixa_ticket(df, p33, p66):
    """Recria Faixa_Ticket com os MESMOS cortes (P33/P66) usados no treino."""
    df = df.copy()
    df["Faixa_Ticket"] = pd.cut(
        df["Total"], bins=[-float("inf"), p33, p66, float("inf")],
        labels=["Baixo", "Médio", "Alto"],
    ).astype(str)
    return df


def preparar(df, p33, p66):
    df = adicionar_faixa_ticket(df, p33, p66)
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes: {faltando}")
    return df[COLUNAS]
