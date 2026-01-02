"""RAG Answer Module - Generoi vastaukset käyttäen Lapua-LLM LoRA-adapteria.

Yhdistää:
1. Hybridi-haun (BM25 + vektori) kontekstin löytämiseen
2. CCG-FAKTUM/lapua-llm-v2 LoRA-adapterin vastauksen generointiin

LoRA-adapteri: https://huggingface.co/CCG-FAKTUM/lapua-llm-v2
Pohjamalli: Qwen/Qwen2.5-1.5B-Instruct
"""

import sys
from dataclasses import dataclass
from typing import Any

# Ensure proper UTF-8 output for Finnish characters
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pipeline.query import SearchResult, TilinpaatosRAG


@dataclass
class RAGAnswer:
    """RAG-vastaus lähteineen."""
    
    question: str
    answer: str
    sources: list[SearchResult]
    model_name: str


class LapuaRAGAssistant:
    """RAG-assistentti Lapuan tilinpäätöstiedoille.
    
    Käyttää:
    - Hybridi-hakua (BM25 + vektori) kontekstin löytämiseen
    - Lapua-LLM LoRA-adapteria vastauksen generointiin
    """
    
    def __init__(
        self,
        year: int,
        lora_model: str = "CCG-FAKTUM/lapua-llm-v2",
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.year = year
        self.lora_model_name = lora_model
        self.base_model_name = base_model
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Initializing Lapua RAG Assistant on {self.device}...")
        
        # Load RAG retriever
        self.rag = TilinpaatosRAG(year, device=self.device)
        
        # Load LLM with LoRA
        self._load_llm(load_in_4bit)
        
        print(f"Lapua RAG Assistant ready!")
    
    def _load_llm(self, load_in_4bit: bool = False) -> None:
        """Load base model with LoRA adapter."""
        print(f"Loading base model: {self.base_model_name}...")
        
        # Use explicit device placement to avoid offloading issues
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        elif self.device == "cuda":
            # Load fully on GPU without offloading
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("cuda")
        else:
            # CPU fallback
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        
        print(f"Loading LoRA adapter: {self.lora_model_name}...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.lora_model_name,
        )
        if self.device == "cuda" and not load_in_4bit:
            self.model = self.model.to("cuda")
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lora_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"LLM loaded successfully!")
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for Lapua-LLM."""
        # Lapua-LLM vastausformaatti: Johtopäätös → Perustelut → Lähteet
        system_prompt = (
            "Olet Lapuan kaupungin tilinpäätösasiantuntija. "
            "LUE KONTEKSTI HUOLELLISESTI ja etsi sieltä TARKAT NUMEROT. "
            "Vastaa kysymyksiin AINOASTAAN kontekstin perusteella. "
            "Anna TARKAT LUVUT euroina tai kappalemäärinä. "
            "Jos kontekstissa lukee '18 kurssia', vastaa '18 kurssia'. "
            "Käytä muotoa: Johtopäätös → Perustelut → Lähteet (sivu)."
        )
        
        user_prompt = f"""Konteksti (tilinpäätöstiedot {self.year}):
{context}

Kysymys: {question}"""
        
        # Qwen chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    def _format_context(self, results: list[SearchResult], max_chars: int = 6000) -> str:
        """Format search results as context string."""
        context_parts = []
        total_chars = 0
        
        for r in results:
            source_info = f"[Sivu {r.page}]" if r.page else ""
            if r.table_id:
                source_info += f" [Taulukko: {r.table_id}]"
            
            chunk_text = f"{source_info}\n{r.text}\n"
            
            if total_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        return "\n---\n".join(context_parts)
    
    def answer(
        self,
        question: str,
        top_k: int = 5,
        max_new_tokens: int = 300,
        temperature: float = 0.3,
    ) -> RAGAnswer:
        """Answer a question using RAG + Lapua-LLM.
        
        Args:
            question: Natural language question
            top_k: Number of context chunks to retrieve
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (lower = more focused)
        
        Returns:
            RAGAnswer with answer text and sources
        """
        # 1. Retrieve relevant context
        results = self.rag.query(question, top_k=top_k, method="hybrid")
        
        if not results:
            return RAGAnswer(
                question=question,
                answer="Ei löytynyt relevanttia tietoa tilinpäätöksestä.",
                sources=[],
                model_name=self.lora_model_name,
            )
        
        # 2. Format context
        context = self._format_context(results)
        
        # 3. Build prompt
        prompt = self._build_prompt(question, context)
        
        # 4. Generate answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return RAGAnswer(
            question=question,
            answer=answer_text.strip(),
            sources=results,
            model_name=self.lora_model_name,
        )
    
    def format_answer(self, rag_answer: RAGAnswer) -> str:
        """Format RAG answer for display."""
        output = [
            f"\n{'='*60}",
            f"KYSYMYS: {rag_answer.question}",
            f"{'='*60}",
            f"\nVASTAUS ({rag_answer.model_name}):\n",
            rag_answer.answer,
            f"\n{'-'*60}",
            f"LÄHTEET:",
        ]
        
        for i, src in enumerate(rag_answer.sources[:3], 1):
            source_info = f"  {i}. "
            if src.page:
                source_info += f"Sivu {src.page}"
            if src.table_id:
                source_info += f" | {src.table_id}"
            output.append(source_info)
        
        return "\n".join(output)


def interactive_mode(year: int) -> None:
    """Interactive Q&A mode."""
    print(f"\n=== Lapua Tilinpäätös {year} RAG + LLM ===")
    print("Kirjoita kysymys tai 'quit' lopettaaksesi.\n")
    
    assistant = LapuaRAGAssistant(year)
    
    while True:
        try:
            question = input("\nKysymys: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        
        answer = assistant.answer(question)
        print(assistant.format_answer(answer))


def main(year: int, question: str | None = None) -> None:
    """Main entry point."""
    if question:
        # Single question mode
        assistant = LapuaRAGAssistant(year)
        answer = assistant.answer(question)
        print(assistant.format_answer(answer))
    else:
        # Interactive mode
        interactive_mode(year)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.rag_answer YEAR [QUESTION]")
        print("       python -m src.pipeline.rag_answer 2024 'Mikä on vuosikate?'")
        sys.exit(1)
    
    year = int(sys.argv[1])
    question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
    main(year, question)

