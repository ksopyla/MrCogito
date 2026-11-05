// How many tokens to train on: Chinchilla-optimal, the ratios well-known small models were trained at, or a fixed count.

export type TokenRule = "custom" | "chinchilla" | "llama3_8b" | "tinyllama" | "smollm2_135m" | "pythia";

export interface TokenRuleSpec {
  id: Exclude<TokenRule, "custom">;
  label: string;
  short: string; // the button text
  source: string;
  perParam?: number; // tokens per parameter
  fixed?: number; // a fixed token count, whatever the size
}

export const CHINCHILLA_PER_PARAM = 20;

export const TOKEN_RULES: TokenRuleSpec[] = [
  { id: "chinchilla", short: "Chinchilla 20×", label: "Chinchilla-optimal: 20 tokens per parameter", perParam: CHINCHILLA_PER_PARAM,
    source: "Hoffmann et al. 2022, compute-optimal training; the project's text-check tiers use about this (0.6B at 30M, 2B at 100M)" },
  { id: "llama3_8b", short: "Llama 3 8B ≈1.9k×", label: "As Llama 3 8B: 15T tokens on 8B parameters", perParam: 15e12 / 8e9,
    source: "Meta, Llama 3 (2024): over-trained for cheap inference" },
  { id: "tinyllama", short: "TinyLlama ≈2.7k×", label: "As TinyLlama: 3T tokens on 1.1B parameters", perParam: 3e12 / 1.1e9,
    source: "Zhang et al. 2024, TinyLlama" },
  { id: "smollm2_135m", short: "SmolLM2-135M ≈15k×", label: "As SmolLM2-135M: 2T tokens on 135M parameters", perParam: 2e12 / 135e6,
    source: "Hugging Face, SmolLM2 (2025)" },
  { id: "pythia", short: "Pythia 300B", label: "As the Pythia suite: 300B tokens at every size", fixed: 300e9,
    source: "Biderman et al. 2023, Pythia" },
];

export const ruleSpec = (id: TokenRule | undefined) => TOKEN_RULES.find((r) => r.id === id);

/** Tokens a rule asks for at this many parameters. */
export function ruleTokens(id: Exclude<TokenRule, "custom">, params: number): number {
  const r = ruleSpec(id)!;
  return r.fixed ?? Math.round((r.perParam ?? CHINCHILLA_PER_PARAM) * params);
}

/** The run's training tokens: its rule applied to the model, or the typed budget. */
export function trainingTokens(run: { token_rule?: TokenRule; token_budget: number }, params: number): number {
  return run.token_rule && run.token_rule !== "custom" && ruleSpec(run.token_rule) ? ruleTokens(run.token_rule, params) : run.token_budget;
}
