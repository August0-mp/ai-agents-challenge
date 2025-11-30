import Groq from "groq-sdk";
import * as fs from "fs";

// Types
interface Message {
  type: string;
  origin: string;
  sender: "CUSTOMER" | "BOT";
  content: string;
  mediaUrl: string | null;
  createdAt: string;
}

interface Conversation {
  conversation_id: string;
  messages: Message[];
}

interface EvalResult {
  conversationId: string;
  messageIndex: number;
  botMessage: string;
  score: number;
  feedback: string;
  context: Message[];
}

interface RuleImprovement {
  ruleName: string;
  originalText: string;
  improvedText: string;
  reason: string;
}

// Config
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY || "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",
});

const MODEL = "llama-3.1-8b-instant";
const DELAY_MS = 100;
const MAX_RETRIES = 3;
const SAMPLE_SIZE = 5;
const USE_SAMPLING = true;

const EVAL_CRITERIA = `
CRITÉRIOS (0-100 pontos):
1. PROGRESSÃO NO FUNIL (30pts) - Avança cliente, não retrocede
2. CLAREZA (25pts) - Uma pergunta por turno, conciso
3. USO DE FERRAMENTAS (25pts) - Chama ferramentas corretas
4. CONVERSÃO (20pts) - Mantém engajamento

PENALIZAÇÕES:
- Link sem frete: -40pts
- Pula prepare_cart: -30pts
- Múltiplas perguntas: -15pts
- Não usa ferramentas: -20pts
`;

// Utils
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

async function callGroq(prompt: string, maxTokens = 300, retries = MAX_RETRIES): Promise<string> {
  for (let i = 1; i <= retries; i++) {
    try {
      const res = await groq.chat.completions.create({
        messages: [{ role: "user", content: prompt }],
        model: MODEL,
        temperature: 0.3,
        max_tokens: maxTokens,
      });
      return res.choices[0]?.message?.content || "{}";
    } catch (err: any) {
      if (err?.status === 429 && i < retries) {
        await sleep(500);
      } else throw err;
    }
  }
  throw new Error("Max retries reached");
}

function loadConversations(path: string): Conversation[] {
  const content = fs.readFileSync(path, "utf-8");
  const lines = content.split(/\r?\n/).map(l => l.trim()).filter(Boolean);

  console.log(`Lendo ${lines.length} linhas`);

  const convos: Conversation[] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    const commaIdx = line.indexOf(",");
    if (commaIdx === -1) continue;

    const id = line.slice(0, commaIdx).trim();
    let msgsRaw = line.slice(commaIdx + 1).trim();

    if (msgsRaw.startsWith('"') && msgsRaw.endsWith('"')) {
      msgsRaw = msgsRaw.slice(1, -1);
    }
    msgsRaw = msgsRaw.replace(/""/g, '"');

    try {
      const messages = JSON.parse(msgsRaw);
      convos.push({ conversation_id: id, messages });
    } catch (e) {
      console.error(`Erro parsing linha ${i}:`, msgsRaw.slice(0, 100));
    }
  }

  console.log(`${convos.length} conversas carregadas\n`);
  return convos;
}

function getContext(msgs: Message[], idx: number): Message[] {
  const start = Math.max(0, idx - 3);
  return msgs.slice(start, idx + 1);
}

function sample<T>(arr: T[], n: number): T[] {
  if (arr.length <= n) return arr;
  return [...arr].sort(() => Math.random() - 0.5).slice(0, n);
}

// Evaluation
function buildEvalPrompt(context: Message[], rules: string): string {
  const convo = context.map(m => `${m.sender}: ${m.content}`).join("\n");
  return `Avalie a última mensagem do BOT (0-100):

CONTEXTO:
${convo}

REGRAS:
${rules}

${EVAL_CRITERIA}

Responda APENAS em JSON (sem markdown):
{"score": <0-100>, "feedback": "<máximo 2 frases curtas>"}`;
}

function collectBotMsgs(convos: Conversation[]) {
  const msgs = [];
  for (const conv of convos) {
    for (let i = 0; i < conv.messages.length; i++) {
      const msg = conv.messages[i];
      if (msg.sender === "BOT") {
        msgs.push({
          conversationId: conv.conversation_id,
          messageIndex: i,
          botMessage: msg.content,
          context: getContext(conv.messages, i),
        });
      }
    }
  }
  return msgs;
}

async function evaluate(convos: Conversation[], rules: string): Promise<EvalResult[]> {
  const botMsgs = collectBotMsgs(convos);
  console.log(`Avaliando ${botMsgs.length} mensagens (delay ${DELAY_MS}ms)\n`);

  const results: EvalResult[] = [];
  const total = botMsgs.length;

  for (let i = 0; i < botMsgs.length; i++) {
    const msg = botMsgs[i];
    const pct = Math.round(((i + 1) / total) * 100);

    try {
      const prompt = buildEvalPrompt(msg.context, rules);
      const raw = await callGroq(prompt, 300);
      const clean = raw.replace(/```json\n?|\n?```/g, '').trim();
      const evaluation = JSON.parse(clean);

      console.log(`[${i + 1}/${total}] (${pct}%) Conv ${msg.conversationId}: ${evaluation.score} - ${evaluation.feedback}`);

      results.push({
        conversationId: msg.conversationId,
        messageIndex: msg.messageIndex,
        botMessage: msg.botMessage,
        score: evaluation.score,
        feedback: evaluation.feedback,
        context: msg.context,
      });

      if (i < botMsgs.length - 1) await sleep(DELAY_MS);
    } catch (err) {
      console.error(`Erro msg ${i + 1}:`, err);
    }
  }

  console.log(`\n✅ ${results.length}/${total} processadas\n`);
  return results;
}

// Analysis
function analyzeLowScores(results: EvalResult[]): EvalResult[] {
  if (!results.length) return [];

  const sorted = [...results].sort((a, b) => a.score - b.score);
  const bottom10pct = sorted.slice(0, Math.max(1, Math.ceil(sorted.length * 0.1)));

  console.log(`\n${bottom10pct.length} msgs no bottom 10%\n`);
  console.log("Top 10 piores:\n");

  bottom10pct.slice(0, 10).forEach((r, i) => {
    console.log(`${i + 1}. Score ${r.score}/100`);
    console.log(`   Conv: ${r.conversationId}`);
    console.log(`   Msg: "${r.botMessage.slice(0, 60)}..."`);
    console.log(`   Issue: ${r.feedback.slice(0, 100)}...\n`);
  });

  return bottom10pct;
}

// Improvement
function buildImprovementPrompt(lowScores: EvalResult[], currentRules: string): string {
  const examples = lowScores.slice(0, 15).map((r, i) => {
    const ctx = r.context.map(m => `${m.sender}: ${m.content}`).join("\n");
    return `EXEMPLO ${i + 1} (Score: ${r.score}/100):
${ctx}
Problema: ${r.feedback}
---`;
  });

  return `Você é especialista em otimizar prompts de chatbots de vendas.

REGRAS ATUAIS:
${currentRules}

PROBLEMAS:
${examples.join("\n\n")}

Analise e proponha melhorias específicas.

DIRETRIZES:
1. Seja específico
2. Mantenha formato
3. Adicione proibições explícitas
4. Reforce pontos críticos
5. Seja conciso

JSON (sem markdown):
{
  "improvements": [
    {
      "ruleName": "<nome>",
      "originalText": "<original>",
      "improvedText": "<melhorado>",
      "reason": "<razão>"
    }
  ],
  "summary": "<resumo>"
}`;
}

async function improveRules(lowScores: EvalResult[], rules: string) {
  const prompt = buildImprovementPrompt(lowScores, rules);
  const raw = await callGroq(prompt, 2000);
  const clean = raw.replace(/```json\n?|\n?```/g, '').trim();
  const result = JSON.parse(clean);

  console.log("Melhorias:");
  console.log(result.summary);
  console.log(`\n${result.improvements.length} regras modificadas\n`);

  return result;
}

function applyImprovements(rules: string, improvements: RuleImprovement[]): string {
  let improved = rules;
  for (const imp of improvements) {
    improved = improved.replace(imp.originalText, imp.improvedText);
  }
  return improved;
}

// Teste
function split(convos: Conversation[]) {
  const shuffled = [...convos].sort(() => Math.random() - 0.5);
  const mid = Math.floor(shuffled.length / 2);
  return {
    control: shuffled.slice(0, mid),
    test: shuffled.slice(mid),
  };
}

function filterByConvos(results: EvalResult[], convos: Conversation[]): EvalResult[] {
  const ids = new Set(convos.map(c => c.conversation_id));
  return results.filter(r => ids.has(r.conversationId));
}

function calcStats(results: EvalResult[]) {
  const scores = results.map(r => r.score).sort((a, b) => a - b);
  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  const median = scores[Math.floor(scores.length / 2)];
  const good = scores.filter(s => s >= 70).length;

  return {
    avgScore: Math.round(avg * 10) / 10,
    medianScore: median,
    goodPct: Math.round((good / scores.length) * 100),
    total: scores.length,
  };
}

async function runTest(allResults: EvalResult[], convos: Conversation[], improvedRules: string) {
  console.log("\nRodando teste...\n");

  const { control, test } = split(convos);
  console.log(`Controle: ${control.length}`);
  console.log(`Teste: ${test.length}\n`);

  const controlResults = filterByConvos(allResults, control);
  const controlStats = calcStats(controlResults);

  console.log("Grupo teste...");
  const testResults = await evaluate(test, improvedRules);
  const testStats = calcStats(testResults);

  const improvement = ((testStats.avgScore - controlStats.avgScore) / controlStats.avgScore) * 100;

  console.log("\nResultados:");
  console.log("Controle:", controlStats);
  console.log("Teste:", testStats);
  console.log(`Melhoria: ${improvement.toFixed(1)}%\n`);

  return { controlStats, testStats, improvement };
}

// Main
async function main() {
  const allConvos = loadConversations("./generated_conversations.csv");
  const rulesModule = await import("./regras");
  const originalRules = rulesModule.rules.storeRules.map(r => r.value.text).join("\n\n");

  console.log(allConvos[0]);
  console.log(originalRules);
  console.log(`${allConvos.length} conversas totais`);

  const convos = USE_SAMPLING ? sample(allConvos, SAMPLE_SIZE) : allConvos;

  // Fase 1: Avaliação
  console.log("FASE 1: AVALIAÇÃO");
  console.log("=".repeat(70));
  const allResults = await evaluate(convos, originalRules);
  const lowScores = analyzeLowScores(allResults);

  // Fase 2: Melhoria
  console.log("FASE 2: MELHORIA");
  console.log("=".repeat(70));
  const { improvements, summary } = await improveRules(lowScores, originalRules);
  const improvedRules = applyImprovements(originalRules, improvements);

  fs.writeFileSync("./regras_melhoradas.txt", improvedRules);
  console.log("✅ Regras salvas em regras_melhoradas.txt");

  // Fase 3: Teste
  console.log("FASE 3: TESTE");
  console.log("=".repeat(70));
  const ab = await runTest(allResults, convos, improvedRules);

  console.log(`\nMelhoria final: ${ab.improvement.toFixed(1)}%\n`);
}

main().catch(console.error);