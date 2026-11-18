"""App Streamlit: aplica o modelo de segmentação a novas vendas."""
import glob

import joblib
import pandas as pd
import streamlit as st

from funcoes import COL_NUM, COLUNAS, preparar

st.set_page_config(page_title="Segmentação de Clientes", page_icon="🛒")


@st.cache_resource
def carregar():
    # Procura qualquer arquivo .joblib na pasta (funciona com ou sem acento no nome)
    arquivos = glob.glob("*.joblib")
    if not arquivos:
        st.error("Arquivo do modelo (.joblib) não encontrado no repositório.")
        st.stop()
    return joblib.load(arquivos[0])


m = carregar()
pipe, p33, p66 = m["pipeline"], m["p33"], m["p66"]

st.title("🛒 Segmentação de Clientes do Supermercado")
st.caption(f"Modelo K-Means com K={m['k']} · Silhouette no treino: {m['silhouette']:.4f} "
           "· O segmento descreve a compra (fatura), não a pessoa.")

aba1, aba2 = st.tabs(["Uma compra", "Arquivo em lote (CSV)"])

with aba1:
    c1, c2 = st.columns(2)
    filial = c1.selectbox("Filial", ["A", "B", "C"])
    tipo = c1.selectbox("Tipo de cliente", ["Membro", "Normal"])
    genero = c1.selectbox("Gênero", ["Feminino", "Masculino"])
    linha = c1.selectbox("Linha de produto", [
        "Saúde e Beleza", "Acessórios Eletrônicos", "Casa e Estilo de Vida",
        "Esportes e Viagem", "Alimentos e Bebidas", "Acessórios de Moda"])
    pagto = c2.selectbox("Pagamento", ["Carteira Digital", "Dinheiro", "Cartão de Crédito"])
    dia = c2.selectbox("Dia da semana", ["Segunda", "Terça", "Quarta", "Quinta",
                                         "Sexta", "Sábado", "Domingo"])
    hora = c2.slider("Hora da compra", 0, 23, 14)
    preco = c2.number_input("Preço unitário", 10.0, 100.0, 50.0)
    qtd = c2.number_input("Quantidade", 1, 10, 3)
    aval = c2.slider("Avaliação do cliente", 1.0, 10.0, 7.0, 0.1)
    if st.button("Descobrir segmento"):
        linha_df = pd.DataFrame([{
            "Filial": filial, "Tipo_Cliente": tipo, "Genero": genero,
            "Linha_Produto": linha, "Pagamento": pagto, "Dia_Semana": dia,
            "Preco_Unitario": preco, "Quantidade": qtd,
            "Total": preco * qtd * 1.05,  # preço x quantidade + imposto de 5%
            "Avaliacao": aval, "Hora": hora}])
        seg = int(pipe.predict(preparar(linha_df, p33, p66))[0])
        st.success(f"Esta compra pertence ao **Segmento {seg}**.")

with aba2:
    st.write("Envie um CSV com as colunas: " + ", ".join(c for c in COLUNAS if c != "Faixa_Ticket"))
    arq = st.file_uploader("Arquivo CSV", type="csv")
    if arq:
        dados = pd.read_csv(arq)
        try:
            dados["Segmento"] = pipe.predict(preparar(dados, p33, p66))
            st.dataframe(dados.head(50))
            st.bar_chart(dados["Segmento"].value_counts().sort_index())
            st.download_button("Baixar resultado (CSV)",
                               dados.to_csv(index=False).encode("utf-8"),
                               "vendas_segmentadas.csv", "text/csv")
        except ValueError as e:
            st.error(str(e))
