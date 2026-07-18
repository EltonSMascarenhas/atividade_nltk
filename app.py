import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
import time  # Importado para dar um leve tempo para a animação do spinner aparecer

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard NLP - 10 Problemáticas",
    page_icon="🤖",
    layout="wide"
)

# --- ANIMAÇÃO: BARQUINHO NO RODAPÉ ---
st.markdown(
    """
    <style>
    .barco-animado {
        position: fixed;
        bottom: 15px;
        left: -60px;
        font-size: 80px;
        z-index: 9999;
        animation: navegar 35s linear infinite;
        user-select: none;
    }

    @keyframes navegar {
        0% { left: -60px; transform: scaleX(1); }
        49% { left: 100%; transform: scaleX(1); }
        50% { left: 100%; transform: scaleX(-1); } /* Vira o barco para voltar */
        99% { left: -60px; transform: scaleX(-1); }
        100% { left: -60px; transform: scaleX(1); }
    }
    </style>
    <div class="barco-animado">⛵</div>
    """,
    unsafe_allow_html=True
)

# Download automático dos recursos do NLTK (Com o fix do punkt_tab!)
@st.cache_resource
def carregar_recursos_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

carregar_recursos_nltk()

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.title("Mapa da Navegação")
st.sidebar.markdown("Escolha uma das atividades abaixo para testar a solução:")

atividades = {
    "Atividade 1: Tokenização de Palavras": "act1",
    "Atividade 2: Frequência de Palavras": "act2",
    "Atividade 3: Detecção de Palavras Negativas": "act3",
    "Atividade 4: Remoção de Stopwords": "act4",
    "Atividade 5: Classificação de Sentimento Simples": "act5",
    "Atividade 6: Direcionamento de Chatbot": "act6",
    "Atividade 7: Palavras Frequentes em Reclamações": "act7",
    "Atividade 8: Classificação de Setor (Suporte vs Financeiro)": "act8",
    "Atividade 9: Remoção de Pontuação e Normalização": "act9",
    "Atividade 10: Análise de Sentimento Combinada": "act10",
}

escolha = st.sidebar.radio("Ir para:", list(atividades.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido para análise automatizada de textos de clientes.")
st.sidebar.caption("Desenvolvido por Elton Mascarenhas.")

# --- CONTEÚDO PRINCIPAL ---
st.title("🤖 Processamento de Linguagem Natural com NLTK")
st.markdown(f"### 📍 {escolha}")
st.markdown("---")

# -------------------------------------------------------------------------
# ATIVIDADE 1
if atividades[escolha] == "act1":
    st.subheader("Problemática:")
    st.write("Uma empresa recebe centenas de mensagens. Ela precisa separar automaticamente o texto em palavras para facilitar a análise.")
    
    texto_padrao = "O suporte técnico da empresa resolveu o meu problema muito rápido hoje."
    texto = st.text_area("Texto de entrada:", texto_padrao)
    
    if st.button("Executar Tokenização"):
        with st.spinner("Tokenizando o texto..."):
            time.sleep(0.6)  # Efeito dramático para a animação
            tokens = word_tokenize(texto)
        
        st.toast("Tokenização concluída!", icon="✨")
        st.success("✨ **Resultado da Tokenização:**")
        st.write(tokens)
        st.metric(label="Total de Tokens gerados", value=len(tokens))

# -------------------------------------------------------------------------
# ATIVIDADE 2
elif atividades[escolha] == "act2":
    st.subheader("Problemática:")
    st.write("Um sistema precisa identificar quais palavras aparecem com mais frequência em avaliações de clientes.")
    
    texto_padrao = "excelente produto excelente atendimento entrega rápida produto muito bom excelente"
    texto = st.text_area("Texto de entrada:", texto_padrao)
    
    if st.button("Contar Frequência"):
        with st.spinner("Calculando distribuição de frequência..."):
            time.sleep(0.6)
            tokens = word_tokenize(texto.lower())
            freq = nltk.FreqDist(tokens)
            df_freq = pd.DataFrame(freq.items(), columns=['Palavra', 'Frequência']).sort_values(by='Frequência', ascending=False)
        
        st.toast("Gráfico gerado com sucesso!", icon="📊")
        st.success("📊 **Resultado da Frequência:**")
        st.dataframe(df_freq, use_container_width=True)
        st.bar_chart(data=df_freq, x='Palavra', y='Frequência')

# -------------------------------------------------------------------------
# ATIVIDADE 3
elif atividades[escolha] == "act3":
    st.subheader("Problemática:")
    st.write("Detectar automaticamente mensagens com palavras negativas para priorizar o suporte ao cliente.")
    
    texto_padrao = "O sistema está apresentando um erro péssimo, a experiência foi muito ruim."
    texto = st.text_area("Texto de entrada:", texto_padrao)
    
    if st.button("Verificar Alertas Negativos"):
        with st.spinner("Escaneando termos críticos..."):
            time.sleep(0.5)
            tokens = word_tokenize(texto.lower())
            palavras_alvo = ["ruim", "péssimo", "erro"]
            encontradas = [word for word in tokens if word in palavras_alvo]
        
        if encontradas:
            st.toast("Atenção! Mensagem perigosa detectada.", icon="🚨")
            st.error(f"🚨 **Alerta de Prioridade!** Palavras negativas detectadas: {list(set(encontradas))}")
        else:
            st.toast("Texto limpo!", icon="✅")
            st.success("✅ Nenhuma palavra estritamente negativa detectada na regra condicional.")

# -------------------------------------------------------------------------
# ATIVIDADE 4
elif atividades[escolha] == "act4":
    st.subheader("Problemática:")
    st.write("Palavras como 'de', 'a', 'o', 'para' não ajudam na interpretação e precisam ser removidas.")
    
    texto_padrao = "Eu comprei um produto para a minha mãe e ela gostou muito do resultado."
    texto = st.text_area("Texto de entrada:", texto_padrao)
    
    if st.button("Remover Stopwords"):
        with st.spinner("Filtrando stopwords em português..."):
            time.sleep(0.6)
            tokens = word_tokenize(texto.lower())
            stops = set(stopwords.words('portuguese'))
            filtrados = [w for w in tokens if w not in stops]
        
        st.toast("Ruídos removidos!", icon="🧹")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Antes ({len(tokens)} termos):**")
            st.write(tokens)
        with col2:
            st.success(f"**Depois ({len(filtrados)} termos relevantes):**")
            st.write(filtrados)

# -------------------------------------------------------------------------
# ATIVIDADE 5
elif atividades[escolha] == "act5":
    st.subheader("Problemática:")
    st.write("Entender rapidamente se comentários de clientes são positivos ou negativos com base em palavras-chave.")
    
    texto_padrao = "O serviço foi ótimo e o atendente foi muito bom, adorei tudo!"
    texto = st.text_area("Texto de entrada:", texto_padrao)
    
    if st.button("Analisar Sentimento (Keywords)"):
        with st.spinner("Computando score de sentimento..."):
            time.sleep(0.5)
            tokens = word_tokenize(texto.lower())
            pos_words = ["bom", "ótimo", "excelente", "adorei", "gostei", "rápido"]
            neg_words = ["ruim", "péssimo", "erro", "atrasou", "pior", "odiei"]
            
            score = 0
            for token in tokens:
                if token in pos_words: score += 1
                elif token in neg_words: score -= 1
            
        st.write("### Resultado da Classificação:")
        if score > 0:
            st.toast("Análise concluída: Sentimento Positivo!", icon="🟢") # <-- Animação discreta aqui
            st.success(f"🟢 **Sentimento Positivo** (Score: {score})")
        elif score < 0:
            st.toast("Análise concluída: Sentimento Negativo.", icon="🔴")
            st.error(f"🔴 **Sentimento Negativo** (Score: {score})")
        else:
            st.toast("Análise concluída: Sentimento Neutro.", icon="🟡")
            st.warning(f"🟡 **Sentimento Neutro** (Score: {score})")

# -------------------------------------------------------------------------
# ATIVIDADE 6
elif atividades[escolha] == "act6":
    st.subheader("Problemática:")
    st.write("Um chatbot precisa identificar palavras-chave para direcionar o cliente para o setor correto.")
    
    texto_padrao = "Quero fazer o pagamento do meu boleto que está com erro."
    texto = st.text_area("Mensagem do Cliente:", texto_padrao)
    
    if st.button("Rotear Mensagem"):
        with st.spinner("Analisando intenção de rota..."):
            time.sleep(0.5)
            tokens = word_tokenize(texto.lower())
            
            setor = "Atendimento Geral"
            if "cancelar" in tokens:
                setor = "Setor de Retenção e Cancelamentos"
            elif "erro" in tokens:
                setor = "Suporte Técnico Especializado"
            elif "pagamento" in tokens or "boleto" in tokens:
                setor = "Departamento Financeiro"
                
        st.toast("Mensagem roteada com sucesso!", icon="🔄")
        st.info(f"💬 **Direcionamento Automático:** Encaminhar cliente para: **{setor}**")

# -------------------------------------------------------------------------
# ATIVIDADE 7
elif atividades[escolha] == "act7":
    st.subheader("Problemática:")
    st.write("Identificar as palavras que mais aparecem em reclamações de clientes para melhoria do produto (Filtrando ruídos).")
    
    texto_padrao = "O app deu erro. Que erro chato! Sempre o mesmo erro na hora de logar no app."
    texto = st.text_area("Texto da Reclamação:", texto_padrao)
    
    if st.button("Analisar Reclamação"):
        with st.spinner("Limpando dados e extraindo focos de reclamação..."):
            time.sleep(0.6)
            tokens = word_tokenize(texto.lower())
            stops = set(stopwords.words('portuguese'))
            palavras_limpas = [w for w in tokens if w.isalnum() and w not in stops]
            freq = nltk.FreqDist(palavras_limpas)
            df_reclamacao = pd.DataFrame(freq.items(), columns=['Palavra-Chave', 'Ocorrências']).sort_values(by='Ocorrências', ascending=False)
        
        st.success("🔍 **Foco do Problema Detectado:**")
        st.dataframe(df_reclamacao, use_container_width=True)

# -------------------------------------------------------------------------
# ATIVIDADE 8
elif atividades[escolha] == "act8":
    st.subheader("Problemática:")
    st.write("Classificar mensagens automaticamente em 'suporte técnico' ou 'financeiro' com base em regras condicionais.")
    
    texto_padrao = "Não consigo acessar minha conta, o sistema dá falha no login."
    texto = st.text_area("Mensagem:", texto_padrao)
    
    if st.button("Classificar Categoria"):
        with st.spinner("Analisando vocabulário departamental..."):
            time.sleep(0.5)
            tokens = word_tokenize(texto.lower())
            termos_tecnicos = ["sistema", "login", "senha", "erro", "bug", "acessar", "falha"]
            termos_financeiros = ["pago", "pagamento", "boleto", "fatura", "dinheiro", "cartão", "reembolso"]
            
            score_tech = sum(1 for t in tokens if t in termos_tecnicos)
            score_fin = sum(1 for t in tokens if t in termos_financeiros)
        
        if score_tech > score_fin:
            st.success("🛠️ Categoria Estimada: **SUPORTE TÉCNICO**")
        elif score_fin > score_tech:
            st.success("💰 Categoria Estimada: **FINANCEIRO**")
        else:
            st.warning("⚖️ Categoria Indefinida (Empate ou termos insuficientes)")

# -------------------------------------------------------------------------
# ATIVIDADE 9
elif atividades[escolha] == "act9":
    st.subheader("Problemática:")
    st.write("Limpar textos removendo pontuação e deixando apenas palavras relevantes em letras minúsculas (Normalização).")
    
    texto_padrao = "Olá!!! O produto chegou... Porém, veio quebrado?! Preciso de ajuda urgente."
    texto = st.text_area("Texto bruto:", texto_padrao)
    
    if st.button("Normalizar Texto"):
        with st.spinner("Higienizando caracteres e pontuações..."):
            time.sleep(0.6)
            texto_lower = texto.lower()
            texto_sem_punc = texto_lower.translate(str.maketrans('', '', string.punctuation))
            resultado_final = word_tokenize(texto_sem_punc)
            
        st.toast("Texto limpo com sucesso!", icon="🧼")
        st.success("🧼 **Texto Normalizado e Limpo:**")
        st.write(resultado_final)

# -------------------------------------------------------------------------
# ATIVIDADE 10
elif atividades[escolha] == "act10":
    st.subheader("Problemática:")
    st.write("Combinar tokenização + condicional para analisar o sentimento básico de avaliações de produtos de forma automatizada.")
    
    texto_padrao = "O produto é incrivelmente maravilhoso, superou todas as expectativas!"
    texto = st.text_area("Avaliação do Produto:", texto_padrao)
    
    if st.button("Rodar Pipeline de Sentimento"):
        with st.spinner("Iniciando pipeline NLP completo..."):
            time.sleep(0.8)
            tokens = word_tokenize(texto.lower())
            positivas = ["maravilhoso", "excelente", "bom", "perfeito", "recomendo", "amei", "lindo"]
            negativas = ["odiei", "ruim", "defeito", "quebrado", "pessimo", "péssimo", "atrasou"]
            
            cont_pos = sum(1 for t in tokens if t in positivas)
            cont_neg = sum(1 for t in tokens if t in negativas)
        
        st.write("### Resultado do Pipeline:")
        if cont_pos > cont_neg:
            st.toast("Pipeline finalizado: Cliente Satisfeito!", icon="🥳") # <-- Animação discreta aqui
            st.success(f"🥳 **Cliente Satisfeito** (Palavras Positivas: {cont_pos} | Negativas: {cont_neg})")
        elif cont_neg > cont_pos:
            st.toast("Pipeline finalizado: Cliente Insatisfeito.", icon="🤬")
            st.error(f"🤬 **Cliente Insatisfeito** (Palavras Positivas: {cont_pos} | Negativas: {cont_neg})")
        else:
            st.toast("Pipeline finalizado: Sentimento Neutro.", icon="😐")
            st.info(f"😐 **Sentimento Neutro / Inconclusivo** (Palavras Positivas: {cont_pos} | Negativas: {cont_neg})")