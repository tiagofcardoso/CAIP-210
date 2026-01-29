# 🐛 AI Assistant - Troubleshooting Guide

## 🔧 Correções Aplicadas

### 1. ✅ Idioma da IA (CORRIGIDO)
**Problema**: IA sempre respondia em PT-BR, mesmo em modo EN

**Solução**:
- Adicionada instrução explícita no prompt: `"IMPORTANTE: Responda SEMPRE em Português (PT-BR)."` ou `"IMPORTANT: Always respond in English."`
- Labels dinâmicos: `Aluno/Student` e `Tutor/Tutor`

**Teste**:
1. Mude para English (toggle EN)
2. Erre uma questão
3. Pergunte à IA
4. Resposta deve vir em inglês

---

### 2. ✅ Microfone e Voz (CORRIGIDO)
**Problema**: Microfone não funcionava, voz não falava

**Soluções Aplicadas**:
- Inicialização segura dos serviços de voz
- Fallback para `currentLanguage`
- Logs de debug adicionados

---

## 🧪 Como Testar

### Teste 1: Idioma da IA
```
1. Acesse https://barbershop-toni.web.app
2. Clique no toggle de idioma (EN)
3. Responda uma questão errada
4. Clique "🤖 Ask AI"
5. Digite: "Explain it more simply"
6. ✅ Resposta deve vir em INGLÊS
```

### Teste 2: Microfone
```
1. Abra o console do navegador (F12)
2. Procure por logs: [Voice] Initializing...
3. Verifique: [Voice] Voice input supported: true
4. Clique "🤖 Ask AI"
5. SEGURE o botão 🎤
6. Fale algo
7. ✅ Texto deve aparecer em tempo real
```

### Teste 3: Voz (Text-to-Speech)
```
1. Abra o console (F12)
2. Clique "🤖 Ask AI"
3. Ative o toggle 🔊 (deve ficar azul/roxo)
4. Digite uma pergunta
5. Procure no console: [VoiceOutput] Speaking in language: pt-BR
6. ✅ Deve ouvir a resposta
```

---

## 🔍 Diagnóstico de Problemas

### Problema: Microfone não funciona

**Passo 1: Verifique o Console**
```javascript
// Abra F12 > Console
// Procure por:
[Voice] Initializing voice services...
[Voice] Voice input supported: true/false
```

**Se `supported: false`:**
- ❌ Navegador não suporta (use Chrome/Edge)
- ❌ Está em HTTP (precisa HTTPS para produção)
- ❌ Firefox tem suporte limitado

**Passo 2: Verifique Permissões**
```
1. Clique no cadeado 🔒 na barra de endereço
2. Verifique se "Microfone" está permitido
3. Se bloqueado, clique e permita
4. Recarregue a página
```

**Passo 3: Teste Manualmente**
```javascript
// Cole no console:
voiceInput.start(
    (text) => console.log('Interim:', text),
    (text) => console.log('Final:', text)
);
// Fale algo
// Deve aparecer no console
```

---

### Problema: Voz não fala

**Passo 1: Verifique o Console**
```javascript
// Procure por:
[VoiceOutput] Speaking in language: pt-BR
```

**Se não aparecer:**
- ❌ Auto-speak está OFF (clique no toggle 🔊)
- ❌ Volume do dispositivo está mudo
- ❌ Navegador bloqueou áudio (precisa interação do usuário primeiro)

**Passo 2: Teste Manualmente**
```javascript
// Cole no console:
voiceOutput.speak('Teste de voz em português');
// Deve falar
```

**Passo 3: Verifique Vozes Disponíveis**
```javascript
// Cole no console:
speechSynthesis.getVoices().forEach(v => 
    console.log(v.name, v.lang)
);
// Deve mostrar lista de vozes
// Procure por: pt-BR ou en-US
```

---

### Problema: IA responde em idioma errado

**Passo 1: Verifique o idioma atual**
```javascript
// Cole no console:
console.log('Current language:', currentLanguage);
// Deve mostrar: 'pt' ou 'en'
```

**Passo 2: Verifique o contexto**
```javascript
// Cole no console:
console.log('AI Context:', aiAssistant.currentContext);
// Verifique se language está correto
```

**Passo 3: Força o idioma**
```javascript
// Para forçar inglês:
currentLanguage = 'en';
// Depois abra o chat novamente
```

---

## 🌐 Compatibilidade de Navegadores

### Chrome/Edge ✅ RECOMENDADO
- ✅ Speech-to-Text: Perfeito
- ✅ Text-to-Speech: Perfeito
- ✅ Vozes: Google voices (alta qualidade)

### Safari (iOS 14.5+) ⚠️ BOM
- ✅ Speech-to-Text: Bom
- ✅ Text-to-Speech: Bom
- ⚠️ Precisa permissão de microfone
- ⚠️ Pode ter delay inicial

### Firefox ❌ LIMITADO
- ⚠️ Speech-to-Text: Limitado
- ✅ Text-to-Speech: Funciona
- ❌ Recomendado usar Chrome/Edge

---

## 📱 Mobile

### Android (Chrome)
- ✅ Tudo funciona perfeitamente
- ⚠️ Precisa permitir microfone
- ⚠️ Precisa HTTPS (produção)

### iOS (Safari)
- ✅ Funciona bem
- ⚠️ Pode ter delay inicial
- ⚠️ Precisa interação do usuário para áudio

---

## 🔒 HTTPS Requirement

**IMPORTANTE**: Para produção, voz requer HTTPS!

**Seu site**: https://barbershop-toni.web.app ✅ (já é HTTPS)

**Localhost**: http://localhost:8000 ✅ (permitido para desenvolvimento)

---

## 🆘 Comandos de Debug

### Verificar tudo de uma vez
```javascript
// Cole no console:
console.log('=== AI ASSISTANT DEBUG ===');
console.log('Language:', currentLanguage);
console.log('Voice Input:', voiceInput?.supported);
console.log('Voice Output:', voiceOutput?.supported);
console.log('AI Context:', aiAssistant?.currentContext);
console.log('Auto-speak:', voiceOutput?.getAutoPlay());
console.log('Available voices:', speechSynthesis.getVoices().length);
```

### Testar fluxo completo
```javascript
// 1. Teste voz
voiceOutput.speak('Teste de voz');

// 2. Teste microfone (fale após executar)
voiceInput.start(
    (text) => console.log('Você disse:', text),
    (text) => console.log('Final:', text)
);

// 3. Teste IA (substitua pela sua pergunta)
aiAssistant.sendMessage('Explique machine learning')
    .then(resp => console.log('IA respondeu:', resp));
```

---

## ✅ Checklist de Verificação

Antes de reportar um bug, verifique:

- [ ] Navegador é Chrome/Edge
- [ ] Site está em HTTPS
- [ ] Permissão de microfone concedida
- [ ] Volume do dispositivo não está mudo
- [ ] Console não mostra erros
- [ ] `currentLanguage` está definido
- [ ] Serviços de voz foram inicializados
- [ ] Testou os comandos de debug acima

---

## 📊 Logs Esperados (Console)

Quando tudo está funcionando:

```
[Voice] Initializing voice services...
[Voice] Current language: pt
[Voice] Voice input supported: true
[Voice] Voice output supported: true
[VoiceInput] Language set to: pt-BR
[VoiceOutput] Speaking in language: pt-BR
```

---

## 🐛 Reportar Bug

Se ainda não funcionar, me envie:

1. **Navegador e versão**
2. **Sistema operacional**
3. **Logs do console** (F12 > Console > copie tudo)
4. **O que você tentou fazer**
5. **O que aconteceu**
6. **Screenshot (se possível)**

---

**Última atualização**: 2026-01-29
**Versão**: 1.1 (com correções de idioma e voz)
