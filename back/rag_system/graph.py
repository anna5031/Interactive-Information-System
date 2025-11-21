from __future__ import annotations

from typing import Callable, Dict, List

from langgraph.graph import END, StateGraph

from .answer_generator import AnswerLLM
from .config import INDEX_CONFIG
from .guardrails import GuardrailLLM
from .indoor_map import load_indoor_map
from .navigation import find_route
from .retriever import InformationRetriever, RecommendationRetriever
from .state import RagState
from .vector_index import LocalVectorIndex


SESSION_END_MESSAGE = "필요한 사항이 있으면 다시 불러 주세요."


class RagWorkflowBuilder:
    def __init__(
        self,
        *,
        vector_index: LocalVectorIndex,
        guardrail: GuardrailLLM | None = None,
        answer_llm: AnswerLLM | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.info_retriever = InformationRetriever(vector_index)
        self.reco_retriever = RecommendationRetriever(vector_index)
        self.guardrail = guardrail or GuardrailLLM()
        self.answer_llm = answer_llm or AnswerLLM()
        self.indoor_map = load_indoor_map()

    def build(self) -> Callable[[RagState], RagState]:
        workflow = StateGraph(RagState)

        async def guardrail_node(state: RagState) -> RagState:
            history_lines = [f"{item['role']}: {item['content']}" for item in state.get("conversation_history", [])]
            verdict = self.guardrail.analyze(state["question"], history_lines)
            print(f"Guardrail verdict: {verdict.json()}")
            sanitized = verdict.sanitized_question.strip() or state["question"]
            if not verdict.allowed:
                abort_message = verdict.unsupported_reason or verdict.reason or "지원하지 않는 질문입니다."
            elif not verdict.is_safe:
                abort_message = verdict.reason or "안전하지 않은 요청입니다."
            else:
                abort_message = ""
            if verdict.should_end_session:
                abort_message = abort_message or SESSION_END_MESSAGE
            session_should_end = bool(verdict.should_end_session or abort_message)
            logs = [
                f"Guardrail verdict: allowed={verdict.allowed}, retry={verdict.needs_retry}, reason={verdict.reason}",
                f"Question classified as {verdict.question_type} ({verdict.classification_reason})",
            ]
            return {
                "sanitized_question": sanitized,
                "needs_retry": verdict.needs_retry,
                "guardrail_reason": verdict.reason or "",
                "abort_message": abort_message,
                "session_should_end": session_should_end,
                "question_type": verdict.question_type or "INFORMATION",
                "processing_log": logs,
            }

        async def retry_node(state: RagState) -> RagState:
            message = "질문을 잘 인식하지 못했어요. 다시 말씀해 주세요."
            return {
                "abort_message": message,
                "needs_retry": True,
                "processing_log": ["사용자에게 재질문을 요청했습니다."],
            }

        async def info_retrieval_node(state: RagState) -> RagState:
            question = state.get("sanitized_question") or state["question"]
            result = self.info_retriever.retrieve(question)
            threshold = result.max_score * INDEX_CONFIG.information_threshold if result.max_score else 0.0
            return {
                "retrieved_documents": result.documents,
                "retrieval_scores": result.scores,
                "retrieval_max_score": result.max_score,
                "processing_log": [
                    f"Information retrieval: top={result.max_score:.3f}, threshold={threshold:.3f}, returned {len(result.documents)} docs."
                ],
            }

        async def recommendation_retrieval_node(state: RagState) -> RagState:
            question = state.get("sanitized_question") or state["question"]
            result = self.reco_retriever.retrieve(question)
            threshold = result.max_score * INDEX_CONFIG.recommendation_threshold if result.max_score else 0.0
            return {
                "retrieved_documents": result.documents,
                "retrieval_scores": result.scores,
                "retrieval_max_score": result.max_score,
                "processing_log": [
                    f"Recommendation retrieval + Ko-Reranker: top={result.max_score:.3f}, threshold={threshold:.3f}, returned {len(result.documents)} docs."
                ],
            }

        async def answer_node(state: RagState) -> RagState:
            question_type = (state.get("question_type") or "INFORMATION").upper()
            documents = state.get("retrieved_documents", [])
            scores = state.get("retrieval_scores", [])
            docs_preview: List[str] = []
            if question_type == "RECOMMENDATION":
                for idx, (doc, score) in enumerate(zip(documents, scores), start=1):
                    source = doc.metadata.get("doc_id") or doc.metadata.get("source") or f"chunk-{idx}"
                    docs_preview.append(
                        f"{idx}. score={score:.3f} source={source}\n{doc.page_content}"
                    )
            else:
                docs_preview = [doc.page_content for doc in documents]
            history_digest = "\n".join(
                f"{entry['role']}: {entry['content']}" for entry in state.get("conversation_history", [])[-6:]
            )
            output = self.answer_llm.generate(
                state["sanitized_question"],
                docs_preview,
                self.indoor_map,
                history_digest,
                question_type=question_type,
            )
            base_answer = output.final_answer.strip()
            logs = [f"Answer generated using {question_type} prompt."]
            nav_request: Dict[str, str] = {}
            if output.needs_navigation:
                logs.append(f"Navigation trigger: {output.navigation_trigger or '요청됨'}")
                if output.destination_room:
                    nav_request["destination"] = output.destination_room
                # if output.origin_room:
                #     nav_request["origin"] = output.origin_room
            else:
                logs.append("Navigation not required.")
            return {
                "answer_text": base_answer,
                "needs_navigation": bool(output.needs_navigation),
                "navigation_request": nav_request if nav_request else {},
                "processing_log": logs,
            }
        
        #todo: origin 바꾸기.
        async def navigation_node(state: RagState) -> RagState:
            request = state.get("navigation_request") or {}
            destination = request.get("destination")
            origin = "예시 시작점"
            nav_result = find_route(origin, destination)
            updated_answer = state.get("answer_text", "")
            if nav_result.get("success"):
                updated_answer = f"{updated_answer}\n\n경로 안내: {nav_result['message']}"
            logs: List[str] = []
            if nav_result.get("success"):
                logs.append("Navigation map generated.")
            else:
                logs.append("Navigation unavailable.")
            return {
                "navigation_payload": nav_result,
                "answer_text": updated_answer,
                "navigation_request": {},
                "processing_log": logs,
            }

        async def finalize_node(state: RagState) -> RagState:
            if state.get("abort_message"):
                answer_text = state["abort_message"]
            else:
                answer_text = state.get("answer_text", "")
            history_entries = [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer_text},
            ]
            return {
                "answer_text": answer_text,
                "conversation_history": history_entries,
                "processing_log": ["Final response ready."],
            }

        def guardrail_router(state: RagState) -> str:
            if state.get("abort_message"):
                return "finalize"
            if state.get("needs_retry"):
                return "speech_retry"
            question_type = (state.get("question_type") or "INFORMATION").upper()
            if question_type == "RECOMMENDATION":
                return "reco_retrieve"
            return "info_retrieve"

        def navigation_router(state: RagState) -> str:
            if state.get("abort_message"):
                return "finalize"
            if state.get("needs_navigation"):
                return "navigation"
            return "finalize"

        workflow.add_node("guardrail", guardrail_node)
        workflow.add_node("speech_retry", retry_node)
        workflow.add_node("info_retrieve", info_retrieval_node)
        workflow.add_node("reco_retrieve", recommendation_retrieval_node)
        workflow.add_node("answer", answer_node)
        workflow.add_node("navigation", navigation_node)
        workflow.add_node("finalize", finalize_node)

        workflow.set_entry_point("guardrail")
        workflow.add_conditional_edges(
            "guardrail",
            guardrail_router,
            {
                "speech_retry": "speech_retry",
                "info_retrieve": "info_retrieve",
                "reco_retrieve": "reco_retrieve",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("speech_retry", "finalize")
        workflow.add_edge("info_retrieve", "answer")
        workflow.add_edge("reco_retrieve", "answer")
        workflow.add_conditional_edges(
            "answer",
            navigation_router,
            {"navigation": "navigation", "finalize": "finalize"},
        )
        workflow.add_edge("navigation", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()
