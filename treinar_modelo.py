"""Treina o modelo final e salva tudo em modelo_segmentacao.joblib.

Uso (no Colab, depois de exportar df_prep do notebook):
    df_prep.to_csv('dados_preparados.csv', index=False)
    !python treinar_modelo.py --k 3
"""
import argparse
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from funcoes import COL_CAT, COL_NUM, preparar

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=3, help="número de clusters")
ap.add_argument("--csv", default="dados_preparados.csv")
args = ap.parse_args()

df = pd.read_csv(args.csv)
p33, p66 = df["Total"].quantile(0.33), df["Total"].quantile(0.66)
X = preparar(df, p33, p66)

pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), COL_CAT),
        ("num", StandardScaler(), COL_NUM),
    ])),
    ("kmeans", KMeans(n_clusters=args.k, random_state=42, n_init=10)),
])
pipe.fit(X)

rotulos = pipe.predict(X)
sil = silhouette_score(pipe.named_steps["prep"].transform(X), rotulos)
print(f"K={args.k} | Silhouette = {sil:.4f} | tamanho dos clusters:")
print(pd.Series(rotulos).value_counts().sort_index().to_string())

# Perfil dos segmentos (use para preencher a Seção 6.4 do relatório)
saida = X.copy()
saida["Segmento"] = rotulos
perfil = saida.groupby("Segmento").agg(
    Tamanho=("Total", "size"),
    Total_medio=("Total", "mean"),
    Avaliacao_media=("Avaliacao", "mean"),
    Hora_media=("Hora", "mean"),
    Linha_Produto_mais_comum=("Linha_Produto", lambda s: s.mode().iat[0]),
    Genero_mais_comum=("Genero", lambda s: s.mode().iat[0]),
    Tipo_Cliente_mais_comum=("Tipo_Cliente", lambda s: s.mode().iat[0]),
    Filial_mais_comum=("Filial", lambda s: s.mode().iat[0]),
    Dia_mais_comum=("Dia_Semana", lambda s: s.mode().iat[0]),
).round(2)
perfil.to_csv("perfil_segmentos.csv")
print("\nPerfil dos segmentos:\n", perfil.to_string())

joblib.dump({"pipeline": pipe, "p33": float(p33), "p66": float(p66),
             "k": args.k, "silhouette": float(sil)}, "modelo_segmentacao.joblib")
print("\nSalvo: modelo_segmentacao.joblib e perfil_segmentos.csv")
