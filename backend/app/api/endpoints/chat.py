"""
Chat API endpoints
"""
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import logging
import os
import asyncio

from app.db.session import get_db
from app.schemas.chat import (
    ChatQueryRequest,
    ChatQueryResponse,
    ConversationCreate,
    Conversation,
    ChatMessage
)
from app.services.query_engine import QueryEngine

router = APIRouter()
logger = logging.getLogger(__name__)
qe = QueryEngine()

# In-memory conversation storage (replace with Redis/DB in production)
conversations: Dict[str, Dict[str, Any]] = {}


@router.post("/query", response_model=ChatQueryResponse)
async def process_chat_query(
    request: ChatQueryRequest,
    db: Session = Depends(get_db)
):
    """Process a chat query using RAG"""
    
    # Get conversation history if conversation_id provided
    conversation_history = []
    if request.conversation_id and request.conversation_id in conversations:
        conversation_history = conversations[request.conversation_id]["messages"]
    
    # Process query
    try:
        query_engine = QueryEngine()
        # attach DB session to the engine instance (safe whether __init__ accepts db or not)
        query_engine.db = db
        response = await query_engine.process_query(
            query=request.query,
            fund_id=request.fund_id,
            conversation_history=conversation_history
        )
    except Exception as e:
        logger.exception("Chat query processing failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

    # Update conversation history
    if request.conversation_id:
        if request.conversation_id not in conversations:
            conversations[request.conversation_id] = {
                "fund_id": request.fund_id,
                "messages": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        
        conversations[request.conversation_id]["messages"].extend([
            {"role": "user", "content": request.query, "timestamp": datetime.utcnow()},
            {"role": "assistant", "content": response["answer"], "timestamp": datetime.utcnow()}
        ])
        conversations[request.conversation_id]["updated_at"] = datetime.utcnow()
    
    return ChatQueryResponse(**response)


@router.post("/conversations", response_model=Conversation)
async def create_conversation(request: ConversationCreate):
    """Create a new conversation"""
    conversation_id = str(uuid.uuid4())
    
    conversations[conversation_id] = {
        "fund_id": request.fund_id,
        "messages": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    return Conversation(
        conversation_id=conversation_id,
        fund_id=request.fund_id,
        messages=[],
        created_at=conversations[conversation_id]["created_at"],
        updated_at=conversations[conversation_id]["updated_at"]
    )


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conv = conversations[conversation_id]
    
    return Conversation(
        conversation_id=conversation_id,
        fund_id=conv["fund_id"],
        messages=[ChatMessage(**msg) for msg in conv["messages"]],
        created_at=conv["created_at"],
        updated_at=conv["updated_at"]
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    
    return {"message": "Conversation deleted successfully"}
