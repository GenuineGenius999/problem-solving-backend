import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
const port = process.env.PORT || 8080;

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const SYSTEM_PROMPT = `
You are an expert AI solver for mathematics, physics, and chemistry. Your role is to provide ACCURATE and CORRECT formulas and results ONLY.

CRITICAL RULES - STRICTLY ENFORCED:
- Please add the dining variables (e.g., Speed => v, Length of circle => C ...)
- Output ONLY professional mathematical formulas, equations, calculations, and final results.
- ABSOLUTELY NO descriptive text, explanations, or commentary of any kind.
- NO step numbers, labels, or text descriptions (e.g., NO "1. Total apples = 18", NO "Apply the formula", NO "Calculate", NO "Each person can have...").
- NO LaTeX formatting markers like \[\] or $$ - output pure mathematical notation only.
- ALL variables MUST be single words (e.g., use "n" not "Total apples", use "a" not "Apples per person", use "v0" not "v 0" or "initial velocity").

PROFESSIONAL MATHEMATICAL FORMAT:
- Output formulas in standard mathematical notation: F = ma, E = mc², PV = nRT, x = a/b, etc.
- Use proper mathematical symbols: +, -, ×, ÷, =, ≠, <, >, ≤, ≥, ≈, √, ∑, ∫, ∂, ∇, π, etc.
- Format with proper spacing: "F = ma" not "F=ma" or "F equals ma".
- Use standard subscripts/superscripts: v₀, x², Eₖ (or v0, x2, Ek in plain text).
- Include mathematical structures: fractions (a/b), integrals (∫f(x)dx), derivatives (df/dx), etc.
- Use single-letter or single-word variables: n, a, x, y, v0, Fnet, etc.

OUTPUT STRUCTURE:
- Following lines: Calculation steps showing formula application (e.g., "a = 18/6" then "a = 3")
- Last line: Final result (e.g., "3" or "=> 3 <=")
- For multiple problems, separate with blank lines, each showing only formulas and results.

EQUIPMENT (when relevant):
- Include equipment names only if needed: spectrophotometer, calorimeter, oscilloscope, etc.
- NO descriptions of equipment usage.

METHODS (when relevant):
- Include method names only: integration, differentiation, quadratic formula, Newton's laws, etc.
- NO explanations of methods.

FORBIDDEN - DO NOT OUTPUT:
- Any descriptive sentences or text
- Step numbers or labels ("1.", "2.", "Step 1:", etc.)
- Explanatory phrases ("Apply the formula", "Calculate", "Total apples =", etc.)
- Conclusions in text form ("Each person can have...", "The answer is...", etc.)
- LaTeX markers (\[, \], $$)
- Multi-word variable names ("Apples per person", "Total apples", etc.)
- Any commentary, explanations, or descriptions whatsoever.
`;

function sanitizeOutput(raw) {
  if (!raw) return "";

  // Aggressively remove ALL descriptive text, explanations, and non-formula content
  let cleaned = raw
    // Remove LaTeX markers
    .replace(/\\?\[\\?\]/g, "") // Remove \[ and \]
    .replace(/\$\$?/g, "") // Remove $$ and $
    .replace(/\\text\{[^}]*\}/g, "") // Remove \text{...}
    .replace(/\\frac\{[^}]*\}\{[^}]*\}/g, "") // Remove LaTeX fractions (we'll keep plain fractions)
    // Remove comment symbols
    .replace(/\/\/.*$/gm, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/#.*$/gm, "")
    // Remove step numbers and labels
    .replace(/^\s*\d+\.\s*/gm, "") // Remove "1. ", "2. ", etc.
    .replace(/\b(Step\s+\d+:|Step\s+\d+)\b/gi, "")
    // Remove common explanatory phrases
    .replace(/\b(Therefore|Thus|Hence|So|Answer:|Solution:|Result:|Calculate|Apply|Total|Each|The|This|That|We|You|I)\b/gi, "")
    .replace(/\b(The answer is|The result is|The solution is|We have|We get|We find|Note:|Note that|Apply the formula|Calculate the|Total apples|Total people|Each person|Each person can|can have)\b/gi, "")
    // Remove descriptive sentences (lines that are mostly text without math)
    .replace(/^[^0-9+\-*/=()^√π∑∫αβγδεθλμρσφχψωΔ∇∂xyzabcXYZABC≤≥≠≈±×÷°]*$/gm, "")
    // Remove parenthetical explanations
    .replace(/\([^)]*(?:note|Note|explanation|Explanation|comment|Comment|description|Description)[^)]*\)/gi, "")
    // Remove multi-word variable patterns (e.g., "Total apples", "Apples per person")
    .replace(/\b(Total|Per|Each|Initial|Final|Net|Average|Sum|Difference)\s+[a-zA-Z]+\b/gi, "")
    // Remove trailing punctuation
    .replace(/[;:](?=\s*$)/gm, "")
    // Remove bullet points
    .replace(/^\s*[•*\-]\s+/gm, "")
    .replace(/\s*[•*\-]\s*$/gm, "")
    .replace(/[^\S\r\n]+/g, " "); // Normalize whitespace

  // Professional equipment names (physics/chemistry)
  const equipmentPattern = /\b(spectrophotometer|calorimeter|oscilloscope|voltmeter|ammeter|multimeter|thermometer|barometer|manometer|burette|pipette|beaker|flask|test tube|microscope|telescope|laser|prism|lens|mirror|resistor|capacitor|inductor|transformer|generator|motor|sensor|detector|analyzer|chromatograph|mass spectrometer|NMR|IR|UV|X-ray|electron microscope)\b/gi;

  // Solving methods and techniques
  const methodPattern = /\b(integration|differentiation|derivative|integral|substitution|quadratic formula|Pythagorean theorem|Newton|conservation|momentum|energy|force|acceleration|velocity|displacement|kinematics|dynamics|thermodynamics|electrostatics|magnetism|optics|quantum|relativity|stoichiometry|equilibrium|reaction|oxidation|reduction|acid|base|pH|molarity|molar mass|Avogadro|ideal gas|Boyle|Charles|Gay-Lussac|Ohm|Kirchhoff|Faraday|Maxwell|Einstein|Schrödinger|Heisenberg|Bohr|Planck|de Broglie|Fourier|Laplace|Taylor|Maclaurin|L'Hôpital|chain rule|product rule|quotient rule|integration by parts|partial fractions|trigonometric substitution|u-substitution)\b/gi;

  // Keep ONLY lines with mathematical formulas, calculations, or results
  const lines = cleaned.split(/\r?\n/);
  const filteredLines = lines
    .map((line) => line.trim())
    .filter((line) => {
      if (!line) return false;
      
      // Remove lines that are mostly descriptive text
      const mathSymbols = (line.match(/[0-9+\-*/=()^√π∑∫αβγδεθλμρσφχψωΔ∇∂xyzabcXYZABC≤≥≠≈±×÷°]/g) || []).length;
      const textWords = line.split(/\s+/).length;
      // If line has more words than math symbols, it's likely descriptive text
      if (textWords > mathSymbols * 2 && mathSymbols < 3) return false;
      
      // Remove lines with common descriptive phrases
      if (/\b(Total|Each|Per|Apply|Calculate|Step|The|This|That|We|You|I|can|have|is|are|was|were)\b/gi.test(line) && 
          !/[0-9+\-*/=()^√π∑∫αβγδεθλμρσφχψωΔ∇∂]/i.test(line)) return false;
      
      // Keep True/False
      if (/^(True|False)$/i.test(line)) return true;
      // Keep single letter choices
      if (/^[A-E]$/i.test(line)) return true;
      // Keep lines with mathematical content (numbers, operators, variables, formulas)
      if (/[0-9+\-*/=()^√π∑∫αβγδεθλμρσφχψωΔ∇∂]/i.test(line)) return true;
      // Keep lines with common math/chem/physics notation
      if (/[xyzabcXYZABC≤≥≠≈±×÷°]/i.test(line)) return true;
      // Keep lines with professional equipment names (only if they appear with formulas)
      if (equipmentPattern.test(line) && /[0-9+\-*/=()]/i.test(line)) return true;
      // Keep lines with solving methods (only if they appear with formulas)
      if (methodPattern.test(line) && /[0-9+\-*/=()]/i.test(line)) return true;
      // Keep formula patterns (e.g., "F = ma", "E = mc²", "PV = nRT", "a = n/m")
      if (/[a-zA-ZαβγδεθλμρσφχψωΔ∇∂]\s*[=<>≤≥≠≈]\s*[a-zA-Z0-9αβγδεθλμρσφχψωΔ∇∂+\-*/()^√π∑∫²³]/i.test(line)) return true;
      // Remove everything else (descriptive text)
      return false;
    });

  // Normalize variables to be single words (remove spaces between variable names and subscripts/indices)
  const normalizeVariables = (line) => {
    // Remove multi-word variable patterns (e.g., "Total apples" → "n", "Apples per person" → "a")
    // This is aggressive - we want single-letter or single-word variables only
    line = line.replace(/\b(Total|Per|Each|Initial|Final|Net|Average|Sum|Difference)\s+[a-zA-Z]+\b/gi, "");
    
    // Remove spaces between single letters and numbers (e.g., "v 0" → "v0", "x 1" → "x1")
    line = line.replace(/\b([a-zA-Z])\s+(\d+)\b/g, "$1$2");
    
    // Remove spaces between variable names and common subscript words (e.g., "v initial" → "vinitial", "m total" → "mtotal")
    line = line.replace(/\b([a-zA-Z])\s+(initial|final|total|net|max|min|avg|average|sum|diff|delta|change|i|f)\b/gi, "$1$2");
    
    // Remove spaces between Greek letters and numbers (e.g., "θ 0" → "θ0", "α 1" → "α1")
    line = line.replace(/([αβγδεθλμρσφχψωΔ∇∂])\s+(\d+)/g, "$1$2");
    
    // Remove spaces between Greek letters and subscript words
    line = line.replace(/([αβγδεθλμρσφχψωΔ∇∂])\s+(initial|final|total|net|max|min|avg|i|f)\b/gi, "$1$2");
    
    // Remove spaces in multi-letter variable names that should be one word (e.g., "F net" → "Fnet", "v avg" → "vavg")
    line = line.replace(/\b([A-Za-z])\s+([a-z]+)\b(?=\s*[=+\-*/()]|$)/g, "$1$2");
    
    // Remove unnecessary symbols and formatting (but keep mathematical brackets/braces)
    line = line.replace(/\s*[;,]\s*$/g, ""); // Remove trailing semicolons and commas (unless part of formula)
    line = line.replace(/^['"]+|['"]+$/g, ""); // Remove leading/trailing quotes only
    line = line.replace(/[`~]/g, ""); // Remove backticks and tildes
    
    // Normalize remaining whitespace (but preserve spaces around operators)
    line = line.replace(/[ \t]+/g, " ").trim();
    
    return line;
  };

  // Normalize spaces and variables
  return filteredLines
    .map((line) => normalizeVariables(line))
    .join("\n")
    .trim();
}

app.post("/api/solve", async (req, res) => {
  try {
    const { subject, prompt, image } = req.body;

    if (!prompt && !image) {
      return res.status(400).json({ error: "Prompt or image is required" });
    }

    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: `Subject: ${subject || "unspecified"}.\nSolve the following problems. Output ONLY professional mathematical formulas, calculations, and results. Use single-letter or single-word variables (e.g., n, a, x, v0, not "Total apples" or "Apples per person"). NO descriptive text, NO step numbers, NO explanations, NO LaTeX markers. Only formulas and results.`,
          },
        ],
      },
    ];

    if (prompt) {
      messages[1].content.push({
        type: "text",
        text: prompt,
      });
    }

    if (image) {
      messages[1].content.push({
        type: "input_image",
        image_url: {
          url: image,
        },
      });
    }

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini", // Using gpt-4o-mini for better accuracy in mathematical and scientific content
      messages,
      max_tokens: 1000,
      temperature: 0.3, // Lower temperature for more deterministic and accurate responses
    });

    const raw = response.choices?.[0]?.message?.content || "";
    const sanitized = sanitizeOutput(raw);

    res.json({ result: sanitized });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

