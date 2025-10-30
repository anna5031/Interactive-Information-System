import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple
from threading import Lock, RLock

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langchain_core.exceptions import OutputParserException
load_dotenv()
from .defense_layers import SandwichDefense, SeparateLLMEvaluation, detect_information_leakage
from .filtering import AdvancedFilteringSystem
from .intent_classifier import IntentClassifier
from .navigation import PathNavigationSystem
from .state import UniversityMEState
from .vector_store import MechanicalEngineeringRAG

logger = logging.getLogger(__name__)

NO_INFORMATION_FALLBACK = "죄송합니다. 현재 보유한 정보로는 해당 내용을 확인할 수 없습니다. 다른 질문을 부탁드립니다."
UNSAFE_RESPONSE_MESSAGE = "죄송합니다. 안전한 응답을 제공할 수 없습니다. 다른 질문을 입력해주세요."
UNSUPPORTED_INTENT_MESSAGE = "요청하신 내용은 지원 대상이 아닙니다. 교수/수업/세미나실 정보만 안내 가능합니다."
MALICIOUS_INTENT_MESSAGE = "안전하지 않은 요청으로 감지되어 처리할 수 없습니다."

RESPONSE_POLICY_KEYWORDS = [
    "시스템 지시",
    "시스템 프롬프트",
    "system prompt",
    "developer message",
    "내부 규칙",
    "비공개 지침",
]

STRUCTURED_OUTPUT_GUIDE = f"""
[Structured Output 규칙]
- 시스템은 RAGStructuredOutput 스키마(자동 파싱)에 따라 응답합니다.
- final_answer: 검색된 정보로 답변을 정리합니다. 정보가 부족하면 반드시 다음 문장을 그대로 사용하세요: "{NO_INFORMATION_FALLBACK}"
- reasoning_steps: 답변 도출 과정을 2~5단계로 요약합니다. 정보가 부족한 경우에도 확인 과정을 기록합니다.
- retrieved_facts: 신뢰 가능한 근거 문장을 bullet 형식으로 최대 5개 기록합니다. 근거가 없으면 빈 리스트를 유지합니다.
- intent가 seminar_recommendation이라면 검색된 모든 세미나실 후보를 비교·평가하여 최적의 선택 이유와 함께 final_answer와 retrieved_facts에 반영하세요.
- missing_information: 부족하거나 추가로 확보해야 하는 정보를 구체적인 항목 리스트로 작성합니다. 정보가 충분하면 빈 리스트를 유지합니다.
- needs_navigation: 사용자의 질문이 위치·경로 안내를 요구한다고 판단될 때만 True로 설정합니다.
- follow_up_questions: 사용자가 이어서 확인하면 좋은 질문을 0~3개 제안합니다.
- destination_room: 경로 안내가 필요한 경우 목적지 방 이름/호수(예: "301")를 입력합니다. 필요 없으면 null을 유지합니다.
- retry_query: 정보가 부족할 때 원래의 질의의 내용과 의도를 유지(검색이 더 잘되도록 원본 질의를 수정하는 것도 가능함)하면서 부족한 정보를 보충할 수 있는 질의를 제안합니다. 필요 없으면 null을 유지합니다.
- 허구 정보를 생성하거나 추측하지 않습니다. 근거가 없으면 final_answer에 위의 기본 메시지를 사용하고 missing_information을 채워주세요.
"""
# - follow_up_questions: 사용자가 이어서 확인하면 좋은 질문을 0~3개 제안합니다.

class RAGStructuredOutput(BaseModel):
    final_answer: str = Field(..., description="사용자에게 전달할 최종 답변")
    reasoning_steps: List[str] = Field(default_factory=list, description="답변을 도출한 핵심 단계")
    retrieved_facts: List[str] = Field(
        default_factory=list, description="출처 문서에서 확인한 근거 문장 또는 요약"
    )
    needs_navigation: bool = Field(False, description="안내 지도가 필요한 경우 True")
    missing_information: List[str] = Field(
        default_factory=list, description="답변에 필요한데 확보되지 않은 정보"
    )
    # follow_up_questions: List[str] = Field(
    #     default_factory=list, description="사용자에게 제안할 후속 질문 또는 확인 사항"
    # )
    destination_room: Optional[str] = Field(
        default=None,
        description="경로 안내가 필요한 경우 목적지 방 이름 또는 호수",
    )
    retry_query: Optional[str] = Field(
        default=None,
        description="정보가 부족할 때 추가 검색할 쿼리",
    )


class GuardrailVerdict(BaseModel):
    is_safe: bool = Field(default=True, description="요청을 처리할 수 있는지 여부")
    reason: str = Field(default="ok", description="판단 근거 요약")
    category: Literal[
        "ok",
        "system_manipulation",
        "personal_information",
        "academic_misconduct",
        "abuse",
        "other",
    ] = Field(default="ok", description="차단 카테고리")


class UniversityMEInfoSystem:
    def __init__(
        self,
        data_path: Optional[Path] = None,
        enable_intent_rewrite: bool = False,
        use_guardrail: bool = False,
    ):
        default_data = Path(__file__).resolve().parents[0] / "n7_professor_lectures_seminar.json"
        self.data_path = data_path or default_data
        self.enable_intent_rewrite = enable_intent_rewrite
        self.use_guardrail = use_guardrail

        self.initialization_error: Optional[str] = None
        self.rag_system = MechanicalEngineeringRAG(auto_initialize=False)
        self._rag_lock: Lock = Lock()
        self._rag_initialized = False

        self.intent_classifier = IntentClassifier(enable_rewrite=self.enable_intent_rewrite)
        self._navigation_system: Optional[PathNavigationSystem] = None
        self._navigation_lock: Lock = Lock()

        self.filtering_system = AdvancedFilteringSystem()
        self.sandwich_defense = SandwichDefense()
        self.llm_evaluation = SeparateLLMEvaluation()

        self._llm_lock: RLock = RLock()
        self._main_llm: Optional[ChatGroq] = None
        self._structured_main_llm = None
        self._intent_guard: Optional[ChatGroq] = None
        self._intent_guard_structured = None

        self.max_question_sentences = 5
        self.max_question_tokens = 900
        self.workflow = self._create_workflow()

    @property
    def navigation_system(self) -> PathNavigationSystem:
        if self._navigation_system is None:
            with self._navigation_lock:
                if self._navigation_system is None:
                    self._navigation_system = PathNavigationSystem()
        return self._navigation_system

    @property
    def main_llm(self) -> ChatGroq:
        if self._main_llm is None:
            with self._llm_lock:
                if self._main_llm is None:
                    self._main_llm = ChatGroq(
                        model_name="openai/gpt-oss-120b",
                        temperature=0.2,
                    )
        return self._main_llm

    @property
    def structured_main_llm(self):
        if self._structured_main_llm is None:
            with self._llm_lock:
                if self._structured_main_llm is None:
                    self._structured_main_llm = self.main_llm.with_structured_output(RAGStructuredOutput)
        return self._structured_main_llm

    @property
    def intent_guard(self) -> ChatGroq:
        if self._intent_guard is None:
            with self._llm_lock:
                if self._intent_guard is None:
                    self._intent_guard = ChatGroq(
                        model_name="llama-3.1-8b-instant",
                        temperature=0,
                    )
        return self._intent_guard

    @property
    def intent_guard_structured(self):
        if self._intent_guard_structured is None:
            with self._llm_lock:
                if self._intent_guard_structured is None:
                    self._intent_guard_structured = self.intent_guard.with_structured_output(GuardrailVerdict)
        return self._intent_guard_structured

    def _initialize_rag(self) -> None:
        if self._rag_initialized:
            return
        with self._rag_lock:
            if self._rag_initialized:
                return

            try:
                existing_records = self.rag_system.db_manager.count()
            except Exception:
                existing_records = 0

            try:
                if existing_records == 0:
                    if not self.data_path.exists():
                        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
                    self.rag_system.initialize_from_json(self.data_path, rebuild=False)
                else:
                    self.rag_system.refresh_documents()
            except Exception as exc:
                self.initialization_error = str(exc)
                logger.exception("RAG 초기화에 실패했습니다.")
                raise
            else:
                self.initialization_error = None
                self._rag_initialized = True
                try:
                    doc_count = self.rag_system.db_manager.count()
                except Exception:
                    doc_count = -1
                if doc_count >= 0:
                    logger.info("RAG 시스템 초기화 완료 - %d개 문서 로드", doc_count)
                else:
                    logger.info("RAG 시스템 초기화 완료")

    async def ensure_rag_ready(self) -> None:
        if self._rag_initialized:
            return
        await asyncio.to_thread(self._initialize_rag)

    def _ensure_llms_ready(self) -> None:
        _ = self.structured_main_llm
        _ = self.intent_guard_structured

    async def warm_up(self, *, include_navigation: bool = False) -> None:
        try:
            await self.ensure_rag_ready()
        except Exception:
            # 오류는 initialization_error에 기록되고 로그로 남으므로 여기서는 무시
            pass
        else:
            await asyncio.to_thread(self.rag_system.db_manager.ensure_embeddings_ready)

        await asyncio.to_thread(self._ensure_llms_ready)

        if include_navigation:
            await asyncio.to_thread(lambda: self.navigation_system)

    def _preprocess_query(self, user_query: str) -> Dict:
        normalized = user_query.strip()
        if not normalized:
            return {"is_valid": False, "message": "질문이 비어 있습니다. 기계공학과 관련 내용을 입력해주세요."}

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?？])\s+", normalized) if sentence.strip()]
        warnings: List[str] = []

        if len(sentences) > self.max_question_sentences:
            normalized = " ".join(sentences[: self.max_question_sentences])
            warnings.append(f"질문이 길어 처음 {self.max_question_sentences}문장만 사용했습니다.")

        tokens = normalized.split()
        if len(tokens) > self.max_question_tokens:
            normalized = " ".join(tokens[: self.max_question_tokens])
            warnings.append("질문 길이 제한으로 일부만 처리했습니다.")

        return {
            "is_valid": True,
            "normalized": normalized,
            "warnings": warnings,
        }

    def _apply_structured_guidelines(self, prompt: str) -> str:
        prompt = prompt.rstrip()
        return f"{prompt}\n\n{STRUCTURED_OUTPUT_GUIDE}\n"

    def _guardrail_screen(self, query: str) -> Dict:
        prompt = f"""
역할: 기계공학과 안내 시스템 보안 감시자
사용자 요청을 검토하여 안전한지 판단하세요.

요청: \"\"\"{query}\"\"\"

JSON으로만 답변:
{{
  "is_safe": true/false,
  "reason": "판단 근거",
  "category": "ok|system_manipulation|personal_information|academic_misconduct|abuse|other"
}}
"""
        try:
            verdict: GuardrailVerdict = self.intent_guard_structured.invoke(prompt)
            result = verdict.dict()
        except Exception:
            result = {"is_safe": True, "reason": "guard_fallback", "category": "ok"}

        return {
            "is_safe": bool(result.get("is_safe", True)),
            "reason": result.get("reason", "guard_fallback"),
            "category": result.get("category", "ok"),
        }

    def _build_rejection(self, message: str, blocked_by: Optional[List[str]] = None) -> Dict:
        return {
            "final_output": message,
            "filter_results": {"is_safe": False, "blocked_by": blocked_by or [], "confidence": 0.95},
            "detected_intent": "blocked",
            "intent_confidence": 0.0,
            "retrieved_documents": [],
            "llm_response": "",
            "needs_navigation": False,
            "navigation_info": {},
            "processing_log": ["Query rejected before processing."],
        }

    def _build_initial_state(
        self,
        question: str,
        question_index: int,
        total_questions: int,
        preprocessing_notes: List[str],
    ) -> UniversityMEState:
        return UniversityMEState(
            user_query=question,
            detected_intent="",
            intent_confidence=0.0,
            retrieved_documents=[],
            backup_documents=[],
            llm_response="",
            final_output="",
            filter_checks={},
            filter_results={},
            llm_evaluation={},
            structured_output={},
            needs_navigation=False,
            navigation_info={},
            processing_log=[f"질문 {question_index}/{total_questions} 처리 시작: {question}"],
            question_index=question_index,
            total_questions=total_questions,
            preprocessing_notes=list(preprocessing_notes),
            sanitized_query="",
            effective_query=question,
            rewritten_query="",
            rewrite_attempted=not self.enable_intent_rewrite,
            llm_prompt_query=question,
            retry_query="",
            llm_retry_used=False,
            intent_is_malicious=False,
            abort_message="",
            needs_retry=False,
            retry_reason="",
        )

    def _retrieve_documents_for_intent(
        self,
        intent: str,
        query: str,
        *,
        top_k: int = 10,
        preview_k: int = 5,
    ) -> Dict[str, List[str]]:
        if intent == "seminar_recommendation":
            docs = self.rag_system.search_all_seminar_rooms(query)
            preview_docs = docs
            backup_docs: List[Any] = []
        else:
            docs = self.rag_system.search(query, top_k=top_k)
            preview_docs = docs[:preview_k]
            backup_docs = docs[preview_k:]
        return {
            "preview": [doc.page_content for doc in preview_docs],
            "backup": [doc.page_content for doc in backup_docs],
            "total": len(docs),
            "preview_count": len(preview_docs),
        }

    def _create_workflow(self):
        """LangGraph 워크플로우 생성"""

        def preprocess_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Preprocessing user query...")
            analysis = self._preprocess_query(state["user_query"])

            if not analysis["is_valid"]:
                state["abort_message"] = analysis["message"]
                state["filter_results"] = {
                    "is_safe": False,
                    "blocked_by": ["pre_filter"],
                    "confidence": 0.95,
                }
                state.setdefault("preprocessing_notes", [])
                state["processing_log"].append(f"Preprocessing rejected query: {analysis['message']}")
                return state

            normalized = analysis.get("normalized", state["user_query"])
            state["sanitized_query"] = normalized
            state["effective_query"] = normalized
            state["llm_prompt_query"] = normalized

            notes = list(analysis.get("warnings") or [])
            if notes:
                state.setdefault("preprocessing_notes", [])
                state["preprocessing_notes"].extend(notes)

            state["processing_log"].append("Preprocessing completed.")
            return state

        def guardrail_node(state: UniversityMEState) -> UniversityMEState:

            state["processing_log"].append("Running guardrail checks...")
            query = state.get("sanitized_query") or state["user_query"]
            verdict = self._guardrail_screen(query)
            state.setdefault("filter_checks", {})["guardrail"] = verdict

            reason = verdict.get("reason")
            if reason and reason not in {"ok", "guard_fallback"}:
                state.setdefault("preprocessing_notes", [])
                state["preprocessing_notes"].append(f"사전 보안 점검 결과: {reason}")

            if not verdict.get("is_safe", True):
                state["abort_message"] = (
                    "죄송합니다. 보안 정책에 따라 해당 질문을 처리할 수 없습니다. 기계공학과 관련된 다른 질문을 입력해주세요."
                )
                state["filter_results"] = {
                    "is_safe": False,
                    "blocked_by": [verdict.get("category", "guardrail")],
                    "confidence": 0.95,
                }
                state["processing_log"].append("Guardrail rejected the query.")
            else:
                state.setdefault("filter_results", {"is_safe": True, "blocked_by": [], "confidence": 1.0})

            return state

        def input_filtering_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append(
                f"Starting input filtering (question {state['question_index']}/{state['total_questions']})..."
            )
            for note in state.get("preprocessing_notes", []):
                state["processing_log"].append(f"Preprocess note: {note}")

            sanitized = self.filtering_system.sanitize_text(state.get("sanitized_query", state["user_query"]))
            state["sanitized_query"] = sanitized
            state["filter_checks"] = {}
            state["filter_results"] = {
                "is_safe": True,
                "blocked_by": [],
                "confidence": 1.0,
                "sanitized_text": sanitized,
                "warnings": [],
            }
            return state

        def basic_filter_node(state: UniversityMEState) -> UniversityMEState:
            result = self.filtering_system.basic_keyword_filter(state.get("sanitized_query", state["user_query"]))
            state.setdefault("filter_checks", {})["basic"] = result
            if not result.get("is_safe", True):
                state["processing_log"].append(f"Basic filter flagged issue: {result.get('reason')}")
            return state

        def pattern_filter_node(state: UniversityMEState) -> UniversityMEState:
            result = self.filtering_system.pattern_filter(state.get("sanitized_query", state["user_query"]))
            state.setdefault("filter_checks", {})["pattern"] = result
            if not result.get("is_safe", True):
                state["processing_log"].append(f"Pattern filter flagged issue: {result.get('reason')}")
            return state

        def semantic_filter_node(state: UniversityMEState) -> UniversityMEState:
            result = self.filtering_system.semantic_filter(state.get("sanitized_query", state["user_query"]))
            state.setdefault("filter_checks", {})["semantic"] = result
            if not result.get("is_safe", True):
                state["processing_log"].append(f"Semantic filter flagged issue: {result.get('reason')}")
            return state

        def filter_merge_node(state: UniversityMEState) -> UniversityMEState:
            checks = state.get("filter_checks", {})
            required = {"basic", "pattern", "semantic"}
            if not required.issubset(checks):
                return state

            merged = self.filtering_system.combine_results(state.get("sanitized_query", state["user_query"]), checks)
            state["filter_results"] = merged
            state["processing_log"].append(
                "Filter results combined: " + ("safe" if merged["is_safe"] else "blocked")
            )
            if not merged["is_safe"]:
                state["abort_message"] = self._get_rejection_message(merged["blocked_by"])
                state["processing_log"].append(
                    f"Filtering blocked query: {merged['blocked_by']} -> {state['abort_message']}"
                )
            return state

        def intent_classification_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Classifying user intent...")
            classification = self.intent_classifier.classify_intent(
                state.get("sanitized_query", state["user_query"])
            )

            state["detected_intent"] = classification["intent"]
            state["intent_confidence"] = classification["confidence"]

            if self.enable_intent_rewrite:
                rewritten_query = (
                    classification.get("rewritten_query") or state.get("sanitized_query", state["user_query"])
                ).strip()
                state["rewritten_query"] = rewritten_query or state.get("sanitized_query", state["user_query"])
                state["rewrite_attempted"] = False
            else:
                state["rewritten_query"] = ""
                state["rewrite_attempted"] = True

            state["effective_query"] = state.get("sanitized_query", state["user_query"])
            state["llm_prompt_query"] = state["effective_query"]
            state["retry_query"] = ""
            state["llm_retry_used"] = False
            state["intent_is_malicious"] = bool(classification.get("is_malicious", False))
            state["processing_log"].append(
                f"Intent: {classification['intent']} (confidence: {classification['confidence']:.2f})"
            )

            allowed_intents = {"professor_info", "class_info", "seminar_recommendation", "multiple"}
            if classification["intent"] not in allowed_intents:
                state["abort_message"] = UNSUPPORTED_INTENT_MESSAGE
            if state["intent_is_malicious"] or classification.get("risk_level") in {"medium", "high"}:
                state["abort_message"] = MALICIOUS_INTENT_MESSAGE

            return state

        def retry_router_node(state: UniversityMEState) -> UniversityMEState:
            mode = state.get("retry_mode", "")
            if not mode:
                return state

            if mode == "llm_query":
                retry_query = state.get("retry_query", "").strip()
                state["processing_log"].append(f"Retry router applied LLM query: '{retry_query}'")
                state["effective_query"] = retry_query
                state["llm_prompt_query"] = retry_query
                state["retry_mode"] = ""
                state["needs_retry"] = True
                return state

            if mode == "intent_query":
                rewritten_query = state.get("rewritten_query", "").strip()
                state["processing_log"].append(f"Retry router applied intent rewrite: '{rewritten_query}'")
                state["effective_query"] = rewritten_query
                state["llm_prompt_query"] = rewritten_query
                state["retry_mode"] = ""
                state["needs_retry"] = True
                return state

            state["retry_mode"] = ""
            return state

        def rag_retrieval_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Retrieving relevant documents...")

            intent = state["detected_intent"]
            query = state.get("effective_query", state["user_query"])

            retrieval = self._retrieve_documents_for_intent(intent, query)

            state["retrieved_documents"] = retrieval["preview"]
            state["backup_documents"] = retrieval["backup"]
            state["llm_prompt_query"] = query
            state["processing_log"].append(
                f"Retrieved {retrieval['total']} documents (primary {retrieval['preview_count']} used) for query '{query}'."
            )

            return state

        def _build_llm_prompt(query: str, context: str) -> str:
            base_prompt = self.sandwich_defense.create_sandwich_prompt(query, context)
            return self._apply_structured_guidelines(base_prompt)

        def llm_generation_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Generating LLM response...")
            state["needs_retry"] = False
            state["retry_reason"] = ""
            state["abort_message"] = ""
            state["retry_mode"] = ""

            prompt_query = state.get("llm_prompt_query", state["user_query"])
            prompt = _build_llm_prompt(prompt_query, "\n".join(state["retrieved_documents"]))
            state["processing_log"].append("Using default multi-layer defense prompt.")

            try:
                structured = self.structured_main_llm.invoke(prompt)
                print("Structured LLM output:", structured)  # --- IGNORE ---
            except OutputParserException as exc:
                state["processing_log"].append(f"Structured LLM output failed: {exc}")
                fallback_text = ""
                try:
                    raw_response = self.main_llm.invoke(prompt)
                    fallback_text = getattr(raw_response, "content", str(raw_response)).strip()
                    state["processing_log"].append("Fallback raw LLM response generated instead.")
                except Exception as gen_exc:
                    state["processing_log"].append(f"Fallback LLM call also failed: {gen_exc}")
                final_answer = fallback_text or NO_INFORMATION_FALLBACK
                structured = RAGStructuredOutput(
                    final_answer=final_answer,
                    reasoning_steps=[],
                    retrieved_facts=[],
                    needs_navigation=False,
                    missing_information=[],
                    destination_room=None,
                    retry_query=None,
                )
            except Exception as exc:
                state["processing_log"].append(f"Unexpected structured LLM error: {exc}")
                raise
            state["structured_output"] = structured.dict()
            state["llm_response"] = structured.final_answer.strip()
            if structured.reasoning_steps:
                state["processing_log"].append(
                    f"Structured reasoning steps provided: {len(structured.reasoning_steps)} 단계"
                )

            missing_info = list(structured.missing_information or [])
            
            retry_query = (structured.retry_query or "").strip()
            state["retry_query"] = retry_query

            if missing_info:
                print("Missing information detected:", missing_info)
                state["processing_log"].append(
                    "Structured output reports missing information: " + ", ".join(missing_info)
                )

                if state.get("backup_documents"):
                    additional_docs = state["backup_documents"][:5]
                    state["retrieved_documents"].extend(additional_docs)
                    state["backup_documents"] = []
                    state["processing_log"].append(
                        f"추가 문서 {len(additional_docs)}개를 확보하여 재생성을 준비합니다."
                    )
                    state["needs_retry"] = True
                    state["retry_mode"] = "add_docs"
                    state["retry_reason"] = "추가 문서로 재생성 재시도"
                    return state

                if (
                    retry_query
                    and not state.get("llm_retry_used", False)
                    and retry_query.lower() != state.get("effective_query", "").lower()
                ):
                    state["processing_log"].append(f"Retrying with LLM suggested query: '{retry_query}'")
                    state["retry_query"] = retry_query
                    state["llm_retry_used"] = True
                    state["needs_retry"] = True
                    state["retry_reason"] = "LLM이 제안한 재검색 쿼리로 재시도"
                    state["retry_mode"] = "llm_query"
                    return state

                rewritten_query = state.get("rewritten_query", "").strip()
                rewrite_available = (
                    self.enable_intent_rewrite
                    and bool(rewritten_query)
                    and rewritten_query.lower() != state.get("effective_query", "").lower()
                    and not state.get("rewrite_attempted", False)
                )

                if rewrite_available:
                    state["processing_log"].append(f"Retrying with rewritten query: '{rewritten_query}'")
                    state["rewrite_attempted"] = True
                    state["needs_retry"] = True
                    state["retry_reason"] = "재작성된 질문으로 검색 재시도"
                    state["retry_mode"] = "intent_query"
                    return state

                state["processing_log"].append("추가 정보를 확보할 수 없어 기본 안내로 종료합니다.")
                state["abort_message"] = NO_INFORMATION_FALLBACK
            if structured.needs_navigation or structured.destination_room:
                state["needs_navigation"] = True
                state["processing_log"].append("Structured output requested navigation assistance.")
            destination_room = structured.destination_room
            if destination_room and destination_room.strip():
                state["processing_log"].append(f"Structured output destination detected: {destination_room}")
                state.setdefault("navigation_info", {})
                state["navigation_info"]["destination_room"] = destination_room.strip()
            return state

        def policy_regex_check_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Running policy regex checks on response...")
            response_text = state.get("llm_response", "") or ""

            filter_checks = state.setdefault("filter_checks", {})
            leak_result = detect_information_leakage(response_text)
            filter_checks["policy_regex_leakage"] = leak_result

            blocked = False
            blocked_reasons: List[str] = []

            if leak_result.get("has_leakage"):
                blocked = True
                blocked_reasons.append("policy_regex_leakage")

            keyword_matches = [kw for kw in RESPONSE_POLICY_KEYWORDS if kw.lower() in response_text.lower()]
            filter_checks["policy_keyword_scan"] = {"matches": keyword_matches}
            if keyword_matches:
                blocked = True
                blocked_reasons.append("policy_keyword_scan")

            if blocked:
                state["processing_log"].append(
                    "Policy regex checks flagged response issues: " + ", ".join(blocked_reasons)
                )
                state["abort_message"] = UNSAFE_RESPONSE_MESSAGE
                filter_results = state.setdefault("filter_results", {"is_safe": True, "blocked_by": []})
                filter_results["is_safe"] = False
                filter_results.setdefault("blocked_by", [])
                blocked_by = set(filter_results.get("blocked_by") or [])
                blocked_by.update(blocked_reasons)
                filter_results["blocked_by"] = list(blocked_by)
            else:
                state["processing_log"].append("Policy regex checks passed.")
            print("regex blocked: ", blocked)  # --- IGNORE ---
            return state

        def llm_evaluation_dispatch_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Dispatching LLM evaluations...")
            return state

        async def llm_security_eval_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Security evaluation in progress...")
            result = await self.llm_evaluation.security_evaluation(state["user_query"], state["llm_response"])
            parts = state.setdefault("llm_evaluation", {})
            parts["security"] = result
            return state

        async def llm_content_eval_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Content evaluation in progress...")
            context = "\n".join(state["retrieved_documents"])
            result = await self.llm_evaluation.content_evaluation(state["user_query"], state["llm_response"], context)
            parts = state.setdefault("llm_evaluation", {})
            parts["content"] = result
            return state

        def llm_evaluation_merge_node(state: UniversityMEState) -> UniversityMEState:
            parts = state.get("llm_evaluation", {})
            required_keys = {"security", "content"}
            if not required_keys.issubset(parts):
                return state
            security_result = parts["security"]
            content_result = parts["content"]
            
            combined = self.llm_evaluation.combine_evaluations([security_result, content_result])
            state["llm_evaluation"] = combined
            state["processing_log"].append("LLM evaluation merged across security/content checks.")
            action = combined.get("recommended_action", "allow")
            if action not in {"allow", "noop"} or not combined.get("is_safe", True):
                state["abort_message"] = UNSAFE_RESPONSE_MESSAGE
            return state

        def navigation_check_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Checking navigation requirements...")
            structured_output = state.get("structured_output", {}) or {}
            destination_hint = (structured_output.get("destination_room") or "").strip()
            structured_hint = bool(structured_output.get("needs_navigation") or destination_hint)
            if destination_hint:
                state.setdefault("navigation_info", {})
                state["navigation_info"]["destination_room"] = destination_hint
            should_offer = structured_hint or self.navigation_system.should_offer_navigation(
                state["detected_intent"],
                state.get("sanitized_query", state["user_query"]),
                state["llm_response"],
            )
            state["needs_navigation"] = should_offer
            if should_offer:
                state["processing_log"].append("Navigation assistance offered")
            return state

        def final_response_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Generating final response...")

            abort_message = (state.get("abort_message") or "").strip()
            if abort_message:
                state["needs_navigation"] = False
                state["final_output"] = abort_message
                state["processing_log"].append("Final response aborted due to policy checks.")
                return state

            structured = state.get("structured_output", {})
            final_response = state.get("llm_response", "").strip() or NO_INFORMATION_FALLBACK

            supplemental_sections: List[str] = []

            missing_info = structured.get("missing_information") or []
            if missing_info:
                formatted = "\n".join(f"- {item}" for item in missing_info)
                supplemental_sections.append(f"🔎 추가 확인 필요\n{formatted}")

            # follow_ups = structured.get("follow_up_questions") or []
            # if follow_ups:
            #     formatted = "\n".join(f"- {item}" for item in follow_ups[:5])
            #     supplemental_sections.append(f"제안 질문\n{formatted}")

            retrieved_facts = structured.get("retrieved_facts") or []
            if retrieved_facts:
                formatted = "\n".join(f"- {fact}" for fact in retrieved_facts[:5])
                supplemental_sections.append(f"📚 근거 메모\n{formatted}")

            destination_room = (structured.get("destination_room") or "").strip()
            if destination_room:
                supplemental_sections.append(f"🗂️ 목적지: {destination_room}")

            nav_info = (state.get("navigation_info") or {}).copy()
            if nav_info.get("success"):
                map_lines = []
                message = (nav_info.get("message") or "").strip()
                if message:
                    map_lines.append(message)
                map_path = (nav_info.get("map_path") or "").strip()
                if map_path:
                    map_lines.append(f"지도 파일: {map_path}")
                if map_lines:
                    supplemental_sections.append("🗺️ 경로 안내\n" + "\n".join(map_lines))
            elif nav_info:
                failure_message = (nav_info.get("message") or "지도 생성 중 오류가 발생했습니다.").strip()
                supplemental_sections.append(f"❌ 지도 생성에 실패했습니다: {failure_message}")

            if supplemental_sections:
                final_response = f"{final_response}\n\n" + "\n\n".join(supplemental_sections)

            if state.get("needs_navigation") and not nav_info.get("success"):
                final_response += "\n\n📍 위치 찾기에 도움이 필요하시면 '경로 안내해주세요'라고 말씀해주세요!"

            state["final_output"] = final_response
            state["processing_log"].append("Final response generated")
            return state

        def navigation_generation_node(state: UniversityMEState) -> UniversityMEState:
            state["processing_log"].append("Generating navigation map...")
            destination = (state.get("navigation_info", {}) or {}).get("destination_room")

            if destination:
                nav_result = self.navigation_system.generate_navigation_map(destination)
                nav_result["destination_room"] = destination
                state["navigation_info"] = nav_result
                if nav_result["success"]:
                    state["processing_log"].append("Navigation map generated successfully.")
                else:
                    state["processing_log"].append(
                        f"Navigation map generation failed: {nav_result.get('message', 'unknown reason')}."
                    )
            else:
                state["processing_log"].append("Navigation destination not provided; skipping map generation.")

            return state

        def should_continue_after_filter(state: UniversityMEState) -> str:
            if state.get("abort_message"):
                return "final_response"
            return "intent_classification"

        def should_continue_after_preprocess(state: UniversityMEState) -> str:
            if state.get("abort_message"):
                return "final_response"
            return "guardrail_check" if self.use_guardrail else "input_filtering"

        def should_continue_after_guardrail(state: UniversityMEState) -> str:
            if state.get("abort_message"):
                return "final_response"
            return "input_filtering"

        def should_continue_after_intent(state: UniversityMEState) -> str:
            if state.get("abort_message"):
                return "final_response"
            return "rag_retrieval"

        def should_retry_after_generation(state: UniversityMEState) -> str:
            if state.get("abort_message"):
                return "final_response"
            if state.get("needs_retry"):
                if state.get("retry_mode") == "add_docs":
                    return "llm_generation"
                return "retry_router"
            return "policy_regex_check"

        def should_proceed_after_policy_regex(state: UniversityMEState) -> str:
            if state.get("abort_message"):
                return "final_response"
            return "llm_evaluation_dispatch"

        def should_proceed_after_evaluation(state: UniversityMEState) -> str:
            if state.get("abort_message"):
                return "final_response"
            return "navigation_check"

        def should_generate_navigation(state: UniversityMEState) -> str:
            destination_present = bool((state.get("navigation_info", {}) or {}).get("destination_room"))
            if state.get("abort_message"):
                return "final_response"
            if state.get("needs_navigation") and destination_present:
                return "navigation_generation"
            return "final_response"

        workflow = StateGraph(UniversityMEState)

        workflow.add_node("query_preprocess", preprocess_node)
        if self.use_guardrail:
            workflow.add_node("guardrail_check", guardrail_node)
        workflow.add_node("input_filtering", input_filtering_node)
        workflow.add_node("basic_filter", basic_filter_node)
        workflow.add_node("pattern_filter", pattern_filter_node)
        workflow.add_node("semantic_filter", semantic_filter_node)
        workflow.add_node("filter_merge", filter_merge_node)
        workflow.add_node("intent_classification", intent_classification_node)
        workflow.add_node("retry_router", retry_router_node)
        workflow.add_node("rag_retrieval", rag_retrieval_node)
        workflow.add_node("llm_generation", llm_generation_node)
        workflow.add_node("policy_regex_check", policy_regex_check_node)
        workflow.add_node("llm_evaluation_dispatch", llm_evaluation_dispatch_node)
        workflow.add_node("llm_security_eval", llm_security_eval_node)
        workflow.add_node("llm_content_eval", llm_content_eval_node)
        workflow.add_node("llm_evaluation_merge", llm_evaluation_merge_node)
        workflow.add_node("navigation_check", navigation_check_node)
        workflow.add_node("navigation_generation", navigation_generation_node)
        workflow.add_node("final_response", final_response_node)

        workflow.set_entry_point("query_preprocess")
        if self.use_guardrail:
            workflow.add_conditional_edges(
                "query_preprocess",
                should_continue_after_preprocess,
                {
                    "guardrail_check": "guardrail_check",
                    "final_response": "final_response",
                },
            )
            workflow.add_conditional_edges(
                "guardrail_check",
                should_continue_after_guardrail,
                {
                    "input_filtering": "input_filtering",
                    "final_response": "final_response",
                },
            )
        else:
            workflow.add_conditional_edges(
                "query_preprocess",
                should_continue_after_preprocess,
                {
                    "input_filtering": "input_filtering",
                    "final_response": "final_response",
                },
            )
        workflow.add_edge("input_filtering", "basic_filter")
        workflow.add_edge("input_filtering", "pattern_filter")
        workflow.add_edge("input_filtering", "semantic_filter")
        workflow.add_edge("basic_filter", "filter_merge")
        workflow.add_edge("pattern_filter", "filter_merge")
        workflow.add_edge("semantic_filter", "filter_merge")
        workflow.add_conditional_edges(
            "filter_merge",
            should_continue_after_filter,
            {
                "intent_classification": "intent_classification",
                "final_response": "final_response",
            },
        )
        workflow.add_conditional_edges(
            "intent_classification",
            should_continue_after_intent,
            {
                "rag_retrieval": "rag_retrieval",
                "final_response": "final_response",
            },
        )
        workflow.add_edge("retry_router", "rag_retrieval")
        workflow.add_edge("rag_retrieval", "llm_generation")
        workflow.add_conditional_edges(
            "llm_generation",
            should_retry_after_generation,
            {
                "retry_router": "retry_router",
                "llm_generation": "llm_generation",
                "policy_regex_check": "policy_regex_check",
                "final_response": "final_response",
            },
        )
        workflow.add_conditional_edges(
            "policy_regex_check",
            should_proceed_after_policy_regex,
            {
                "llm_evaluation_dispatch": "llm_evaluation_dispatch",
                "final_response": "final_response",
            },
        )
        workflow.add_edge("llm_evaluation_dispatch", "llm_security_eval")
        workflow.add_edge("llm_evaluation_dispatch", "llm_content_eval")
        workflow.add_edge("llm_content_eval", "llm_evaluation_merge")
        workflow.add_edge("llm_security_eval", "llm_evaluation_merge")
        workflow.add_conditional_edges(
            "llm_evaluation_merge",
            should_proceed_after_evaluation,
            {
                "navigation_check": "navigation_check",
                "final_response": "final_response",
            },
        )
        workflow.add_conditional_edges(
            "navigation_check",
            should_generate_navigation,
            {"navigation_generation": "navigation_generation", "final_response": "final_response"},
        )
        workflow.add_edge("navigation_generation", "final_response")
        workflow.add_edge("final_response", END)

        return workflow.compile()

    def _get_rejection_message(self, blocked_by: List[str]) -> str:
        if "basic_filter" in blocked_by:
            return "부적절한 키워드가 포함된 요청입니다. 기계공학과 관련 질문만 가능합니다."
        if "pattern_filter" in blocked_by:
            return "시스템 보안상 처리할 수 없는 요청입니다."
        if "semantic_filter" in blocked_by:
            return "기계공학과와 관련 없는 질문입니다. 교수, 수업, 세미나실 정보만 안내 가능합니다."
        return "요청을 처리할 수 없습니다. 적절한 질문을 해주세요."

    async def process_query(self, user_query: str) -> Dict:
        if not self._rag_initialized or self.initialization_error:
            try:
                await self.ensure_rag_ready()
            except Exception:
                logger.warning("RAG 초기화 시도가 실패했습니다. initialization_error=%s", self.initialization_error)

        if self.initialization_error:
            return self._build_rejection(
                f"시스템 초기화 오류가 발생했습니다: {self.initialization_error}",
                blocked_by=["rag_initialization"],
            )

        question = (user_query or "").strip()
        initial_state = self._build_initial_state(
            question,
            question_index=1,
            total_questions=1,
            preprocessing_notes=[],
        )
        result_state = await self.workflow.ainvoke(initial_state)
        return result_state
