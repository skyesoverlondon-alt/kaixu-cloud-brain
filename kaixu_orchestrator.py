#!/usr/bin/env python3
# save as: kaixu_orchestrator.py
# Run on same or separate server from Cloud Brain
# Python 3.11+ required

import asyncio
import json
import os
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiohttp
import uvicorn

# =============== CONFIGURATION ===============
KAIXU_BRAIN_URL = os.getenv("KAIXU_BRAIN_URL", "http://localhost:8000")
KAIXU_BRAIN_API_KEY = os.getenv("KAIXU_BRAIN_API_KEY", "kaixu-internal-key")
EXTERNAL_PROVIDERS = {
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "model": "deepseek-chat"
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": "gpt-4o-mini"
    }
}
ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "8080"))

# =============== DATA MODELS ===============
class NBEPContract(BaseModel):
    artifacts_requested: List[str] = Field(default_factory=list)
    format_constraints: List[str] = Field(default_factory=list)
    scope: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)

class IIPFlags(BaseModel):
    iip_mode: str = "none"  # none, facts, contested, research
    require_evidence: bool = False
    require_sources: bool = False
    confidence_threshold: float = 0.7

class PTXConfig(BaseModel):
    primary: str = "kaixu_cloud_brain_v1"
    alts: List[str] = Field(default_factory=list)
    cross_check: bool = False
    transparency: bool = True

class KaixuMetadata(BaseModel):
    nbep_contract: Optional[NBEPContract] = None
    iip_flags: Optional[IIPFlags] = None
    ptx_config: Optional[PTXConfig] = None
    session_id: str = Field(default_factory=lambda: f"sesh_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}")
    user_id: str = "kaixu_operator"

class ChatMessage(BaseModel):
    role: str  # system, user, assistant
    content: str
    timestamp: float = Field(default_factory=time.time)

class ChatCompletionRequest(BaseModel):
    model: str = "kaixu-orchestrator"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    stream: bool = False
    metadata: Optional[KaixuMetadata] = None

class ProviderResponse(BaseModel):
    provider: str
    response: str
    raw_response: Dict[str, Any]
    processing_time: float
    tokens_used: int
    was_filtered: bool = False
    filter_reason: Optional[str] = None
    confidence: float = 1.0

class NBEPExecutionReport(BaseModel):
    contract_summary: str
    commitments: List[str]
    exclusions: List[str]
    execution_plan: List[str]
    completed_artifacts: List[str] = Field(default_factory=list)
    missing_components: List[str] = Field(default_factory=list)
    violation_detected: bool = False
    violation_reason: Optional[str] = None

# =============== ORCHESTRATOR CORE ===============
class KaixuOrchestrator:
    def __init__(self):
        self.session = None
        self.logger = self._setup_logging()
        self.nbep_history = {}
        self.ptx_cache = {}
        
    def _setup_logging(self):
        logger = logging.getLogger("kaixu_orchestrator")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler("/tmp/kaixu_orchestrator.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger
    
    async def ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    # =============== NBEP ENFORCEMENT ===============
    def analyze_nbep_contract(self, request: ChatCompletionRequest) -> NBEPExecutionReport:
        """Parse and validate NBEP contract from request"""
        report = NBEPExecutionReport(
            contract_summary="",
            commitments=[],
            exclusions=[],
            execution_plan=[],
            completed_artifacts=[],
            missing_components=[],
            violation_detected=False
        )
        
        if not request.metadata or not request.metadata.nbep_contract:
            report.contract_summary = "No explicit NBEP contract specified. Defaulting to complete answer."
            return report
        
        contract = request.metadata.nbep_contract
        
        # Analyze artifacts requested
        artifact_map = {
            "full_html_file": "Complete HTML document with inline CSS/JS",
            "multi_section_spec": "Detailed specification with all sections",
            "production_code": "Production-ready code with error handling",
            "complete_solution": "End-to-end working solution",
            "api_endpoint": "Fully implemented API endpoint",
            "database_schema": "Complete database schema with migrations"
        }
        
        artifacts_desc = []
        for artifact in contract.artifacts_requested:
            if artifact in artifact_map:
                artifacts_desc.append(artifact_map[artifact])
                report.commitments.append(f"Deliver {artifact_map[artifact]}")
        
        # Analyze constraints
        constraints_desc = []
        for constraint in contract.format_constraints:
            if constraint == "single_code_block":
                constraints_desc.append("Output in single code block without commentary")
                report.commitments.append("Format as single code block")
            elif constraint == "no_placeholder":
                constraints_desc.append("No TODO/FIXME/IMPLEMENT placeholders")
                report.commitments.append("Eliminate all placeholders")
            elif constraint == "error_handling":
                constraints_desc.append("Include error handling")
                report.commitments.append("Implement error handling")
        
        # Build summary
        report.contract_summary = f"NBEP CONTRACT: You requested {', '.join(artifacts_desc)} with constraints: {', '.join(constraints_desc)}. Scope: {', '.join(contract.scope)}"
        
        # Check for potential violations
        user_message = next((m.content for m in request.messages if m.role == "user"), "")
        violation_indicators = [
            ("partial", ["part of", "basic example", "simplified version"]),
            ("skeleton", ["skeleton code", "template", "framework"]),
            ("placeholder", ["TODO", "FIXME", "IMPLEMENT", "ADD HERE"])
        ]
        
        for violation_type, indicators in violation_indicators:
            if any(indicator.lower() in user_message.lower() for indicator in indicators):
                report.violation_detected = True
                report.violation_reason = f"User specifically requested no {violation_type} implementations"
                report.exclusions.append(f"No {violation_type} implementations")
        
        # Generate execution plan
        if "full_html_file" in contract.artifacts_requested:
            report.execution_plan.extend([
                "1. Parse requirements for HTML structure",
                "2. Create complete HTML5 doctype",
                "3. Add head section with meta tags and title",
                "4. Create inline CSS styles in <style> tag",
                "5. Create inline JavaScript in <script> tag",
                "6. Build complete body with semantic elements",
                "7. Test for browser compatibility",
                "8. Output as single code block"
            ])
        
        return report
    
    def format_nbep_response(self, content: str, report: NBEPExecutionReport) -> str:
        """Format final response with NBEP compliance"""
        if not report.violation_detected:
            header = f"""=== KAIXU NBEP EXECUTION REPORT ===
{report.contract_summary}

COMMITMENTS:
{chr(10).join(f'• {c}' for c in report.commitments)}

EXECUTION PLAN:
{chr(10).join(report.execution_plan)}

DELIVERABLES:
"""
            return f"{header}\n\n{content}\n\n=== END NBEP DELIVERY ==="
        else:
            return f"""=== KAIXU NBEP VIOLATION DETECTED ===

VIOLATION: {report.violation_reason}

I cannot proceed with this request as it would violate the No-Bullshit Execution Protocol.

Instead, I will:
1. Acknowledge the violation
2. Explain why the requested approach would be insufficient
3. Offer alternative complete solutions

Please reformulate your request without asking for partial implementations.
"""
    
    # =============== PROVIDER MANAGEMENT ===============
    async def call_kaixu_brain(self, messages: List[Dict], temperature: float, max_tokens: int) -> ProviderResponse:
        """Call the local Kaixu Cloud Brain v1"""
        start_time = time.time()
        
        try:
            await self.ensure_session()
            
            # Prepare messages for Kaixu Brain
            brain_messages = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    brain_messages.append({"role": msg.role, "content": msg.content})
                else:
                    brain_messages.append(msg)
            
            # Add Kaixu system prompt
            system_prompt = """You are Kaixu Cloud Brain v1, an 8B parameter AI running on private GPU infrastructure.
Your directives:
1. Provide complete, production-ready solutions
2. Never use TODO/FIXME/IMPLEMENT placeholders
3. Include error handling and validation
4. Format code as single blocks when requested
5. Be precise and factual
6. Acknowledge limitations explicitly"""
            
            if not any(m.get("role") == "system" for m in brain_messages):
                brain_messages.insert(0, {"role": "system", "content": system_prompt})
            
            async with self.session.post(
                f"{KAIXU_BRAIN_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {KAIXU_BRAIN_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "kaixu-brain-v1",
                    "messages": brain_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.95
                },
                timeout=60
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    tokens = result.get("usage", {}).get("total_tokens", 0)
                    
                    return ProviderResponse(
                        provider="kaixu_cloud_brain_v1",
                        response=content,
                        raw_response=result,
                        processing_time=time.time() - start_time,
                        tokens_used=tokens,
                        was_filtered=False,
                        confidence=0.95
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"Kaixu Brain error {response.status}: {error_text}")
                    
        except Exception as e:
            self.logger.error(f"Kaixu Brain call failed: {e}")
            return ProviderResponse(
                provider="kaixu_cloud_brain_v1",
                response=f"Kaixu Brain unavailable: {str(e)}",
                raw_response={"error": str(e)},
                processing_time=time.time() - start_time,
                tokens_used=0,
                was_filtered=False,
                confidence=0.0
            )
    
    async def call_external_provider(self, provider: str, messages: List[Dict], temperature: float) -> ProviderResponse:
        """Call external provider (DeepSeek, OpenAI, etc.)"""
        if provider not in EXTERNAL_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        
        config = EXTERNAL_PROVIDERS[provider]
        if not config["api_key"]:
            return ProviderResponse(
                provider=provider,
                response=f"{provider} API key not configured",
                raw_response={"error": "API key missing"},
                processing_time=0,
                tokens_used=0,
                was_filtered=True,
                filter_reason="API key not configured"
            )
        
        start_time = time.time()
        
        try:
            await self.ensure_session()
            
            async with self.session.post(
                config["url"],
                headers={
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": config["model"],
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2048
                },
                timeout=30
            ) as response:
                result = await response.json()
                
                # Check for provider filtering
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                was_filtered = self._detect_provider_filtering(content, provider)
                
                return ProviderResponse(
                    provider=provider,
                    response=content,
                    raw_response=result,
                    processing_time=time.time() - start_time,
                    tokens_used=result.get("usage", {}).get("total_tokens", 0),
                    was_filtered=was_filtered,
                    filter_reason="Safety policy" if was_filtered else None,
                    confidence=0.9
                )
                
        except Exception as e:
            self.logger.error(f"{provider} call failed: {e}")
            return ProviderResponse(
                provider=provider,
                response=f"{provider} error: {str(e)}",
                raw_response={"error": str(e)},
                processing_time=time.time() - start_time,
                tokens_used=0,
                was_filtered=False,
                confidence=0.0
            )
    
    def _detect_provider_filtering(self, content: str, provider: str) -> bool:
        """Detect if provider filtered/refused the request"""
        filter_indicators = {
            "openai": [
                "I cannot", "I'm unable", "as an AI", "I don't generate",
                "against my policy", "content guidelines", "I apologize"
            ],
            "deepseek": [
                "I cannot", "I'm sorry", "我不", "根据我的", "guidelines"
            ]
        }
        
        content_lower = content.lower()
        indicators = filter_indicators.get(provider, [])
        
        # Check if response is suspiciously short or generic
        if len(content.strip()) < 50:
            return True
        
        # Check for filter phrases
        for indicator in indicators:
            if indicator.lower() in content_lower:
                return True
        
        return False
    
    # =============== PTX CROSS-REFERENCE ===============
    async def cross_reference_responses(self, responses: List[ProviderResponse]) -> Dict:
        """Compare responses from multiple providers"""
        if len(responses) < 2:
            return {"agreement": 1.0, "conflicts": [], "summary": "Single provider used"}
        
        # Simple similarity check
        from difflib import SequenceMatcher
        
        contents = [r.response for r in responses if r.response and not r.was_filtered]
        if len(contents) < 2:
            return {"agreement": 0.0, "conflicts": ["Insufficient responses for comparison"], "summary": "Comparison not possible"}
        
        similarities = []
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                similarity = SequenceMatcher(None, contents[i], contents[j]).ratio()
                similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Identify conflicts
        conflicts = []
        if avg_similarity < 0.7:
            conflicts.append(f"Provider responses differ significantly (similarity: {avg_similarity:.2f})")
        
        # Check for filtering
        filtered = [r.provider for r in responses if r.was_filtered]
        if filtered:
            conflicts.append(f"Providers filtered response: {', '.join(filtered)}")
        
        return {
            "agreement": avg_similarity,
            "conflicts": conflicts,
            "summary": f"{len(contents)} providers responded, {len(filtered)} filtered",
            "filtered_providers": filtered
        }
    
    # =============== MAIN ORCHESTRATION ===============
    async def orchestrate_completion(self, request: ChatCompletionRequest) -> Dict:
        """Main orchestration method with NBEP/IIP/PTX enforcement"""
        start_time = time.time()
        session_id = request.metadata.session_id if request.metadata else "unknown"
        
        self.logger.info(f"Orchestrating request for session {session_id}")
        
        # Step 1: NBEP Analysis
        nbep_report = self.analyze_nbep_contract(request)
        
        # Step 2: Determine provider strategy
        primary_provider = "kaixu_cloud_brain_v1"
        alt_providers = []
        
        if request.metadata and request.metadata.ptx_config:
            if request.metadata.ptx_config.primary:
                primary_provider = request.metadata.ptx_config.primary
            if request.metadata.ptx_config.alts:
                alt_providers = request.metadata.ptx_config.alts
            if request.metadata.ptx_config.cross_check:
                alt_providers = list(EXTERNAL_PROVIDERS.keys())
        
        # Step 3: Call primary provider
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Add NBEP context to messages
        if nbep_report.contract_summary:
            nbep_context = f"NBEP Context: {nbep_report.contract_summary}\n\nCommitments: {chr(10).join(nbep_report.commitments)}\n\n"
            if messages[0]["role"] == "system":
                messages[0]["content"] = nbep_context + messages[0]["content"]
            else:
                messages.insert(0, {"role": "system", "content": nbep_context})
        
        primary_response = await self.call_kaixu_brain(
            request.messages,
            request.temperature,
            request.max_tokens
        )
        
        # Step 4: Call alternative providers if requested
        alt_responses = []
        if alt_providers and request.metadata and request.metadata.ptx_config:
            tasks = []
            for provider in alt_providers:
                if provider in EXTERNAL_PROVIDERS:
                    tasks.append(self.call_external_provider(provider, messages, request.temperature))
            
            if tasks:
                alt_responses = await asyncio.gather(*tasks, return_exceptions=True)
                alt_responses = [r for r in alt_responses if not isinstance(r, Exception)]
        
        # Step 5: Cross-reference if multiple providers
        ptx_analysis = await self.cross_reference_responses([primary_response] + alt_responses)
        
        # Step 6: Format final response with NBEP
        final_content = self.format_nbep_response(primary_response.response, nbep_report)
        
        # Step 7: Add PTX transparency if enabled
        if request.metadata and request.metadata.ptx_config and request.metadata.ptx_config.transparency:
            transparency_note = f"""

=== PTX TRANSPARENCY REPORT ===
Primary Provider: {primary_response.provider}
Response Time: {primary_response.processing_time:.2f}s
Tokens Used: {primary_response.tokens_used}
Confidence: {primary_response.confidence:.2f}

"""
            if alt_responses:
                transparency_note += "Alternative Providers:\n"
                for alt in alt_responses:
                    status = "FILTERED" if alt.was_filtered else "OK"
                    transparency_note += f"• {alt.provider}: {status} ({alt.processing_time:.2f}s, {alt.tokens_used} tokens)\n"
                
                if ptx_analysis.get("conflicts"):
                    transparency_note += f"\nConflicts Detected: {', '.join(ptx_analysis['conflicts'])}"
            
            final_content += transparency_note
        
        # Step 8: Log diagnostics
        diagnostics = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time,
            "nbep_report": nbep_report.dict(),
            "ptx_analysis": ptx_analysis,
            "primary_provider": primary_response.provider,
            "primary_tokens": primary_response.tokens_used,
            "primary_time": primary_response.processing_time,
            "alt_providers_called": [r.provider for r in alt_responses],
            "filtered_providers": [r.provider for r in alt_responses if r.was_filtered]
        }
        
        self.logger.info(f"Request completed in {diagnostics['processing_time']:.2f}s")
        
        # Save diagnostics
        with open(f"/home/kaixu/kaixu-brain/logs/diagnostics_{session_id}.json", "w") as f:
            json.dump(diagnostics, f, indent=2)
        
        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-{session_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "kaixu-orchestrator-v1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": primary_response.tokens_used,
                "completion_tokens": len(final_content.split()),
                "total_tokens": primary_response.tokens_used + len(final_content.split())
            },
            "kaixu_diagnostics": diagnostics
        }

# =============== FASTAPI APPLICATION ===============
app = FastAPI(title="Kaixu Orchestrator", version="1.0.0")
orchestrator = KaixuOrchestrator()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "Kaixu Orchestrator v1",
        "status": "operational",
        "brain_url": KAIXU_BRAIN_URL,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "health": "/health",
            "diagnostics": "/diagnostics/{session_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Kaixu Brain
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{KAIXU_BRAIN_URL}/v1/models", timeout=5) as resp:
                brain_ok = resp.status == 200
    except:
        brain_ok = False
    
    return {
        "status": "healthy" if brain_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "orchestrator": "operational",
            "kaixu_brain": "operational" if brain_ok else "unavailable"
        }
    }

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    """Main OpenAI-compatible endpoint with Kaixu protocols"""
    try:
        result = await orchestrator.orchestrate_completion(request)
        return JSONResponse(content=result)
    except Exception as e:
        orchestrator.logger.error(f"Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/diagnostics/{session_id}")
async def get_diagnostics(session_id: str):
    """Retrieve diagnostics for a session"""
    filepath = f"/home/kaixu/kaixu-brain/logs/diagnostics_{session_id}.json"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=404, detail="Diagnostics not found")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=ORCHESTRATOR_PORT,
        log_level="info",
        access_log=True
    )

