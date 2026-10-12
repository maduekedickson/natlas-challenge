from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from api.config import ADAPTER_VERSION
from api.db.session import get_session
from api.services.batch_service import process_csv

router = APIRouter()


@router.post("/batch")
async def batch_process(file: UploadFile = File(...), db: Session = Depends(get_session)):
    contents = await file.read()
    return process_csv(db, contents, adapter_version=ADAPTER_VERSION, filename=file.filename)
