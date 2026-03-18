from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict
import secrets
from database import db

# Initialize the API Router with the /api prefix
api_router = APIRouter(prefix="/api")

# --- Pydantic Schemas for Request Validation ---

class PartnerApplySchema(BaseModel):
    """Schema for new partner applications."""
    company_name: str
    admin_email: str

# --- Partner & Admin Endpoints ---

@api_router.post("/partners/apply")
async def apply_partner(payload: PartnerApplySchema):
    """
    Creates a new Partner record in the database with a PENDING status for the API key.
    """
    try:
        # Check if the email is already registered to prevent duplicates
        existing = await db.partner.find_unique(where={"admin_email": payload.admin_email})
        if existing:
            raise HTTPException(status_code=400, detail="Administrator email already registered.")

        # Create the partner record using Prisma
        new_partner = await db.partner.create(
            data={
                "company_name": payload.company_name,
                "admin_email": payload.admin_email,
                "api_key": f"PENDING_{secrets.token_hex(4)}" # Temporary unique pending key
            }
        )
        return {"status": "success", "partner_id": new_partner.id, "message": "Application received."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@api_router.get("/admin/partners")
async def get_all_partners():
    """
    Returns a list of all registered partners for administrative review.
    """
    partners = await db.partner.find_many(order={"created_at": "desc"})
    return partners

@api_router.post("/admin/partners/{partner_id}/approve")
async def approve_partner(partner_id: str = Path(...)):
    """
    Generates a secure 32-character API key and updates the partner record.
    """
    # Generate a cryptographically secure hex token
    secure_key = f"sk_{secrets.token_hex(16)}"

    try:
        # Update the partner record with the new API key
        updated_partner = await db.partner.update(
            where={"id": partner_id},
            data={"api_key": secure_key}
        )
        
        if not updated_partner:
            raise HTTPException(status_code=404, detail="Partner not found.")

        return {
            "status": "approved",
            "company_name": updated_partner.company_name,
            "api_key": secure_key
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve partner: {str(e)}")

# --- Dashboard & Analytics Endpoints ---

@api_router.get("/dashboard/{partner_id}/stats")
async def get_partner_stats(partner_id: str = Path(...)):
    """
    Provides a breakdown of log counts and classification methods for the partner dashboard.
    """
    # 1. Get total log count
    total_logs = await db.log.count(where={"partner_id": partner_id})

    # 2. Get breakdown by classification method
    # We query for specific strings used in our classification pipeline
    methods = ["Regex", "BERT", "Gemini", "Hybrid (Regex/BERT/Gemini)"]
    breakdown = {}

    for method in methods:
        count = await db.log.count(
            where={
                "partner_id": partner_id,
                "classification_method": {"contains": method}
            }
        )
        breakdown[method] = count

    return {
        "partner_id": partner_id,
        "total_logs_processed": total_logs,
        "method_breakdown": breakdown
    }

@api_router.get("/dashboard/{partner_id}/logs")
async def get_partner_logs(partner_id: str = Path(...)):
    """
    Retrieves the 50 most recent logs for a specific partner.
    """
    logs = await db.log.find_many(
        where={"partner_id": partner_id},
        take=50,
        order={"timestamp": "desc"}
    )
    return logs

@api_router.get("/dashboard/{partner_id}/incidents")
async def get_partner_incidents(partner_id: str = Path(...)):
    """
    Retrieves all AI-generated Incident Reports (RCAs) for a specific partner.
    """
    incidents = await db.incidentreport.find_many(
        where={"partner_id": partner_id},
        order={"created_at": "desc"}
    )
    return incidents
