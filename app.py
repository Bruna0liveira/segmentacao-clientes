"""App Streamlit: aplica o modelo de segmentação a novas vendas."""
import glob

import joblib
import pandas as pd
import streamlit as st

from funcoes import COLUNAS, preparar

st.set_page_config(page_title="Segmentação de Clientes", page_icon="🛒")

IMPOSTO = 1.05  # preço x quantidade + imposto de 5%

LINHAS = [
    "Saúde e Beleza", "Acessórios Eletrônicos", "Casa e Estilo de Vida",
    "Esportes e Viagem", "Alimentos e Bebidas", "Acessórios de Moda"]


@st.cache_resource
def carregar():
    # Procura qualquer arquivo .joblib na pasta (funciona com ou sem acento no nome)
    arquivos = glob.glob("*.joblib")
    if not arquivos:
        st.error("Arquivo do modelo (.joblib) não encontrado no repositório.")
        st.stop()
    return joblib.load(arquivos[0])


# Nomes e ações dos segmentos (valem para o modelo K=3 treinado pelo grupo, Estratégia 3/4).
# Se retreinar o modelo, a numeração pode mudar: confira o perfil_segmentos.csv e ajuste aqui.
SEGMENTOS = {
    0: ("Cliente Premium",
        "Programa de recompensas exclusivo para membros (pontos dobrados aos sábados, "
        "acesso antecipado a lançamentos) e campanhas digitais de produtos de alto valor."),
    1: ("Cliente Econômico",
        "Campanha de adesão ao programa de fidelidade com benefício na primeira compra como membro; "
        "kits e 'compre 2 leve 3' para aumentar o ticket; promoções às terças-feiras."),
    2: ("Cliente Regular",
        "Cashback para migrar do dinheiro para carteira digital ou cartão; lista de compras "
        "recorrente em Alimentos e Bebidas; oferta de membro para migrar ao segmento Premium."),
}


def nome_segmento(n):
    return SEGMENTOS.get(int(n), (f"Segmento {int(n)}", ""))[0]


def acao_segmento(n):
    return SEGMENTOS.get(int(n), ("", ""))[1]


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
    pagto = c2.selectbox("Pagamento", ["Carteira Digital", "Dinheiro", "Cartão de Crédito"])
    dia = c2.selectbox("Dia da semana", ["Segunda", "Terça", "Quarta", "Quinta",
                                         "Sexta", "Sábado", "Domingo"])
    hora = c2.slider("Hora da compra", 0, 23, 14)
    aval = st.slider("Avaliação do cliente", 1.0, 10.0, 7.0, 0.1)

    st.markdown("**Itens da compra** — adicione quantas linhas quiser (use a linha vazia no "
                "final da tabela) e selecione uma linha + Delete para remover.")
    itens_inicial = pd.DataFrame({
        "Linha_Produto": ["Alimentos e Bebidas"],
        "Preco_Unitario": [50.0],
        "Quantidade": [3]})
    itens = st.data_editor(
        itens_inicial,
        num_rows="dynamic",
        key="itens_compra",
        column_config={
            "Linha_Produto": st.column_config.SelectboxColumn(
                "Linha de produto", options=LINHAS, required=True),
            "Preco_Unitario": st.column_config.NumberColumn(
                "Preço unitário", min_value=10.0, max_value=100.0, step=0.5, required=True),
            "Quantidade": st.column_config.NumberColumn(
                "Quantidade", min_value=1, max_value=10, step=1, required=True),
        })

    if st.button("Descobrir segmento"):
        itens = itens.dropna()
        if itens.empty:
            st.warning("Adicione pelo menos um item completo à compra.")
        else:
            itens = itens.assign(Subtotal=itens["Preco_Unitario"] * itens["Quantidade"])
            comuns = {"Filial": filial, "Tipo_Cliente": tipo, "Genero": genero,
                      "Pagamento": pagto, "Dia_Semana": dia,
                      "Avaliacao": aval, "Hora": hora}

            # 1) Segmento de cada item (cada linha vira uma "mini-compra")
            por_item = pd.DataFrame([{
                **comuns,
                "Linha_Produto": r.Linha_Produto,
                "Preco_Unitario": r.Preco_Unitario,
                "Quantidade": int(r.Quantidade),
                "Total": r.Subtotal * IMPOSTO} for r in itens.itertuples()])
            segs_item = pipe.predict(preparar(por_item, p33, p66))

            # 2) Segmento da compra inteira (itens somados em uma única fatura)
            qtd_total = int(itens["Quantidade"].sum())
            subtotal = float(itens["Subtotal"].sum())
            linha_principal = itens.groupby("Linha_Produto")["Subtotal"].sum().idxmax()
            compra = pd.DataFrame([{
                **comuns,
                "Linha_Produto": linha_principal,  # categoria que mais pesou no valor
                "Preco_Unitario": subtotal / qtd_total,  # preço médio ponderado
                "Quantidade": qtd_total,
                "Total": subtotal * IMPOSTO}])
            seg = int(pipe.predict(preparar(compra, p33, p66))[0])

            st.success(f"Compra completa (R$ {subtotal * IMPOSTO:,.2f} com imposto, "
                       f"{qtd_total} unidades): **Segmento {seg}: {nome_segmento(seg)}**.")
            if acao_segmento(seg):
                st.info(f"Ação de marketing sugerida: {acao_segmento(seg)}")

            resumo = itens[["Linha_Produto", "Preco_Unitario", "Quantidade"]].copy()
            resumo["Total c/ imposto"] = (itens["Subtotal"] * IMPOSTO).round(2)
            resumo["Segmento do item"] = [f"{s} - {nome_segmento(s)}" for s in segs_item]
            st.write("Segmento de cada item, analisado isoladamente:")
            st.dataframe(resumo, hide_index=True)

            if qtd_total > 10 or subtotal * IMPOSTO > 1050:
                st.caption("⚠️ Essa compra passa dos valores vistos no treino (até 10 unidades "
                           "e cerca de R$ 1.050 por fatura), então o resultado do segmento da "
                           "compra completa é menos confiável.")

with aba2:
    st.write("Envie um CSV com as colunas: " + ", ".join(c for c in COLUNAS if c != "Faixa_Ticket"))
    arq = st.file_uploader("Arquivo CSV", type="csv")
    if arq:
        dados = pd.read_csv(arq)
        try:
            dados["Segmento"] = pipe.predict(preparar(dados, p33, p66))
            dados["Nome_Segmento"] = dados["Segmento"].map(nome_segmento)
            st.dataframe(dados.head(50))
            st.bar_chart(dados["Nome_Segmento"].value_counts())
            st.download_button("Baixar resultado (CSV)",
                               dados.to_csv(index=False).encode("utf-8"),
                               "vendas_segmentadas.csv", "text/csv")
        except ValueError as e:
            st.error(str(e))
